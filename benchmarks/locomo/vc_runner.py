"""VC pipeline for LocOMo: ingest conversation once, query per question."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

from benchmarks.longmemeval.cost import BudgetTracker
from .dataset import LoCoMoConversation, LoCoMoQuestion

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _sessions_to_messages(conv: LoCoMoConversation) -> list[Message]:
    """Convert LocOMo sessions into flat Message list for VC ingestion.

    Maps speaker_a -> "user", speaker_b -> "assistant".
    Embeds speaker names in content. Merges consecutive same-role turns.
    """
    messages: list[Message] = []

    for session in conv.sessions:
        for i, turn in enumerate(session.turns):
            role = "user" if turn["speaker"] == conv.speaker_a else "assistant"
            text = turn.get("text", "")
            speaker = turn["speaker"]

            if i == 0:
                content = f"[Session from {session.date_time}] {speaker}: {text}"
            else:
                content = f"{speaker}: {text}"

            # Merge consecutive same-role turns
            if messages and messages[-1].role == role:
                messages[-1] = Message(
                    role=role,
                    content=messages[-1].content + "\n" + content,
                )
            else:
                messages.append(Message(role=role, content=content))

    # Ensure even count (VC needs user/assistant pairs)
    if len(messages) % 2 != 0:
        pad_role = "assistant" if messages[-1].role == "user" else "user"
        messages.append(Message(role=pad_role, content=""))

    return messages


def _format_full_haystack(conv: LoCoMoConversation) -> str:
    """Format full conversation for raw-history fallback (uncompacted stores)."""
    parts = [
        f"Below is a conversation between two people: {conv.speaker_a} and {conv.speaker_b}.",
        "The conversation takes place over multiple days and the date of each "
        "conversation is written at the beginning of the conversation.",
        "",
    ]
    for session in conv.sessions:
        parts.append(f"DATE: {session.date_time}")
        parts.append("CONVERSATION:")
        for turn in session.turns:
            parts.append(f"{turn['speaker']}: {turn.get('text', '')}")
        parts.append("")

    return "\n".join(parts)


def _format_question_prompt(
    question: LoCoMoQuestion, conv: LoCoMoConversation, *, explain: bool = False,
) -> str:
    """Build question text for the reader, with category-specific handling."""
    base = f"Based on the conversation between {conv.speaker_a} and {conv.speaker_b}"

    if question.category == 5:
        # Adversarial: randomized (a)/(b)
        random.seed(hash(question.question_id))
        adv = question.adversarial_answer
        if random.random() < 0.5:
            option_a = "Not mentioned in the conversation"
            option_b = adv
        else:
            option_a = adv
            option_b = "Not mentioned in the conversation"
        return (
            f"{base}, answer the following question.\n\n"
            f"Question: {question.question} "
            f"Select the correct answer: (a) {option_a} (b) {option_b}.\n"
            f"If no information is available to answer the question, write "
            f"'Not mentioned in the conversation'.\n"
            f"Short answer:"
        )

    if explain:
        return (
            f"{base}, answer the following question. "
            f"First give your answer, then explain your reasoning by citing "
            f"specific evidence from the conversation.\n\n"
            f"Question: {question.question}\n"
            f"Answer and reasoning:"
        )

    return (
        f"{base}, write a short, direct answer to the following question. "
        f"Give only the answer — no explanation, no hedging, no alternatives.\n\n"
        f"Question: {question.question}\nShort answer:"
    )


def _build_vc_config(
    context_window: int = 65536,
    storage_dir: str | None = None,
    session_id: str = "",
    tagger_provider: str = "openrouter",
    tagger_model: str = "xiaomi/mimo-v2-flash",
    summarizer_provider: str | None = None,
    summarizer_model: str | None = None,
    tagger_mode: str = "split",
    fact_provider: str | None = None,
    fact_model: str | None = None,
    curation_enabled: bool = False,
    curation_provider: str | None = None,
    curation_model: str | None = None,
    supersession: bool = False,
) -> dict:
    """Build a VC config dict for LocOMo benchmark use."""
    chosen_summarizer_provider = summarizer_provider or tagger_provider
    chosen_summarizer_model = summarizer_model or tagger_model

    providers: dict[str, dict] = {}
    if tagger_provider == "anthropic" or chosen_summarizer_provider == "anthropic":
        providers["anthropic"] = {
            "type": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": tagger_model if tagger_provider == "anthropic" else chosen_summarizer_model,
        }
    if tagger_provider == "openai" or chosen_summarizer_provider == "openai":
        providers["openai"] = {
            "type": "generic_openai",
            "base_url": "https://api.openai.com/v1",
            "model": tagger_model if tagger_provider == "openai" else chosen_summarizer_model,
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }
    if tagger_provider == "openrouter" or chosen_summarizer_provider == "openrouter":
        providers["openrouter"] = {
            "type": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": tagger_model if tagger_provider == "openrouter" else chosen_summarizer_model,
            "api_key_env": "OPENROUTER_API_KEY",
        }
    if tagger_provider == "ollama_native" or chosen_summarizer_provider == "ollama_native":
        providers["ollama_native"] = {
            "type": "ollama_native",
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "model": tagger_model if tagger_provider == "ollama_native" else chosen_summarizer_model,
            "num_predict": 500,
            "force_json": True,
        }
    # Fact provider
    if fact_provider and fact_provider not in providers:
        if fact_provider == "openrouter":
            providers["openrouter"] = {
                "type": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": fact_model or tagger_model,
                "api_key_env": "OPENROUTER_API_KEY",
            }
    # Curation provider
    chosen_curation_provider = curation_provider or chosen_summarizer_provider
    chosen_curation_model = curation_model or chosen_summarizer_model
    if chosen_curation_provider and chosen_curation_provider not in providers:
        if chosen_curation_provider == "anthropic":
            providers["anthropic"] = {
                "type": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": chosen_curation_model,
            }
        elif chosen_curation_provider == "openrouter":
            providers["openrouter"] = {
                "type": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": chosen_curation_model,
                "api_key_env": "OPENROUTER_API_KEY",
            }

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
            "provider": tagger_provider,
            "model": tagger_model,
            "max_tags": 10,
            "min_tags": 5,
            "prompt_mode": "detailed",
            "temporal_heuristic_enabled": False,
            "context_bleed_threshold": 0,
            "tagger_mode": tagger_mode,
            **({"fact_provider": fact_provider} if fact_provider else {}),
            **({"fact_model": fact_model} if fact_model else {}),
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
            "provider": chosen_summarizer_provider,
            "model": chosen_summarizer_model,
            "max_tokens": 500,
            "temperature": 0.3,
        },
        "curation": {
            "enabled": curation_enabled,
            "provider": chosen_curation_provider,
            "model": chosen_curation_model,
            "max_response_tokens": 2048,
        },
        "supersession": {
            "enabled": supersession,
            "provider": chosen_summarizer_provider,
            "model": chosen_summarizer_model,
            "batch_size": 25,
        },
        "providers": providers,
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
            "inbound_tagger_type": "embedding",
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
                "anthropic": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
                "openai": {"input_per_1k": 0.00025, "output_per_1k": 0.00200},
            },
        },
    }

    if session_id:
        cfg["session_id"] = session_id
    return cfg


def _clear_compaction_state(cache_dir: str, conv_id: str) -> None:
    """Clear compacted segments and tag summaries but keep TurnTagIndex."""
    import sqlite3
    db_path = Path(cache_dir) / "store.db"
    if not db_path.exists():
        logger.warning("No DB found at %s — nothing to clear", db_path)
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM segment_tags")
    conn.execute("DELETE FROM segments")
    conn.execute("DELETE FROM tag_summaries")
    for tbl in ("segments_fts", "segments_fts_full", "chunk_embeddings"):
        try:
            conn.execute(f"DELETE FROM {tbl}")
        except sqlite3.OperationalError:
            pass
    conn.execute("UPDATE engine_state SET compacted_through = 0")
    conn.commit()
    conn.close()
    logger.info("VC [%s]: cleared compaction state, kept TurnTagIndex", conv_id)


def _dump_payload_log(
    *,
    question_id: str,
    cache_dir: Path,
    question_text: str,
    gold_answer: str,
    category: int,
    tags_matched: list[str],
    tokens_injected: int,
    budget_breakdown: dict,
    assembled_total_tokens: int,
    assembled_budget_breakdown: dict,
    tool_calls: list[dict],
    continuation_count: int,
    stop_reason: str,
    raw_requests: list[dict],
    raw_responses: list[dict],
    hypothesis: str,
    reader_input: int,
    reader_output: int,
    timings: dict,
    cached: bool,
) -> None:
    """Dump payload log for post-run analysis."""
    http_conversation = []
    for i in range(max(len(raw_requests), len(raw_responses))):
        if i < len(raw_requests):
            http_conversation.append({"step": i + 1, "direction": "REQUEST", "body": raw_requests[i]})
        if i < len(raw_responses):
            http_conversation.append({"step": i + 1, "direction": "RESPONSE", "body": raw_responses[i]})

    payload = {
        "_description": "Raw HTTP payloads for LocOMo question.",
        "question_id": question_id,
        "question": question_text,
        "gold_answer": gold_answer,
        "category": category,
        "hypothesis": hypothesis,
        "cached": cached,
        "summary": {
            "tags_matched": tags_matched,
            "tokens_injected": tokens_injected,
            "budget_breakdown": budget_breakdown,
            "assembled_total_tokens": assembled_total_tokens,
            "assembled_budget_breakdown": assembled_budget_breakdown,
            "tool_calls_count": len(tool_calls),
            "continuation_count": continuation_count,
            "stop_reason": stop_reason,
            "tools_used": [tc["tool"] for tc in tool_calls],
            "reader_input_tokens": reader_input,
            "reader_output_tokens": reader_output,
            "timings": timings,
        },
        "tool_calls": tool_calls,
        "http_conversation": http_conversation,
    }

    # Chain analytics
    from benchmarks.longmemeval.chain_analysis import analyze_tool_chain
    payload["chain_analysis"] = analyze_tool_chain(tool_calls)

    payloads_dir = cache_dir / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = payloads_dir / f"{question_id}_{ts}.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as e:
        logger.warning("Failed to save payload log for %s: %s", question_id, e)


def ingest_conversation(
    conv: LoCoMoConversation,
    *,
    context_window: int = 65536,
    fresh: bool = False,
    recompact: bool = False,
    tagger_provider: str = "openrouter",
    tagger_model: str = "xiaomi/mimo-v2-flash",
    summarizer_provider: str | None = None,
    summarizer_model: str | None = None,
    tagger_mode: str = "split",
    fact_provider: str | None = None,
    fact_model: str | None = None,
    cache_dir: Path | None = None,
    supersession: bool = False,
    curation_enabled: bool = False,
    curation_provider: str | None = None,
    curation_model: str | None = None,
) -> tuple[VirtualContextEngine, list[Message], dict]:
    """Ingest a conversation into VC. Returns (engine, messages, stats).

    If cache exists, returns cached engine without re-ingesting.
    """
    base_cache = cache_dir or DEFAULT_CACHE_DIR
    conv_cache = base_cache / conv.conv_id
    if fresh and conv_cache.exists():
        shutil.rmtree(conv_cache)
        logger.info("VC [%s]: cleared cache (--fresh)", conv.conv_id)
    conv_cache.mkdir(parents=True, exist_ok=True)
    storage_dir = str(conv_cache)

    if recompact and not fresh:
        _clear_compaction_state(storage_dir, conv.conv_id)

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"locomo-{conv.conv_id}",
        tagger_provider=tagger_provider,
        tagger_model=tagger_model,
        summarizer_provider=summarizer_provider,
        summarizer_model=summarizer_model,
        tagger_mode=tagger_mode,
        fact_provider=fact_provider,
        fact_model=fact_model,
        curation_enabled=curation_enabled,
        curation_provider=curation_provider,
        curation_model=curation_model,
        supersession=supersession,
    )
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    messages = _sessions_to_messages(conv)
    n_pairs = len(messages) // 2

    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._engine_state.compacted_through > 0
    tags_only = not fully_cached and n_index_entries > 0

    stats: dict = {"conv_id": conv.conv_id, "turns_ingested": 0, "compaction_events": 0}

    if fully_cached:
        logger.info(
            "VC [%s]: CACHE HIT — %d turns indexed, compacted_through=%d",
            conv.conv_id, n_index_entries, engine._engine_state.compacted_through,
        )
        stats["turns_ingested"] = n_index_entries
        stats["compaction_events"] = -1
        stats["cached"] = True
    elif tags_only:
        logger.info(
            "VC [%s]: RECOMPACT — %d turns indexed, re-running compaction",
            conv.conv_id, n_index_entries,
        )
        stats["turns_ingested"] = n_index_entries
        stats["cached"] = False
    else:
        logger.info(
            "VC [%s]: %d messages (%d pairs), ~%dK tokens",
            conv.conv_id, len(messages), n_pairs, conv.est_tokens // 1000,
        )
        stats["cached"] = False

        # Ingest
        t0 = time.time()
        _ingest_start = time.time()

        def _progress(done: int, total: int, entry: object) -> None:
            if done % 20 == 0 or done == total:
                elapsed = time.time() - _ingest_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info("  Ingest: %d/%d (%.1f turns/s, ETA %.0fs)", done, total, rate, eta)

        turns = engine.ingest_history(messages, progress_callback=_progress)
        stats["turns_ingested"] = turns
        logger.info("VC [%s]: ingested %d turns in %.1fs", conv.conv_id, turns, time.time() - t0)

    # Compact
    if not fully_cached:
        t0 = time.time()
        compaction_events = 0
        while True:
            report = engine.compact_manual(messages)
            if report is None:
                break
            compaction_events += 1
            logger.info(
                "  Compact round %d: %d segments, %d tokens freed",
                compaction_events,
                report.segments_compacted,
                report.tokens_freed,
            )
        stats["compaction_events"] = compaction_events
        logger.info("VC [%s]: compaction done (%d rounds) in %.1fs",
                     conv.conv_id, compaction_events, time.time() - t0)

        # Supersession
        if supersession:
            telem = engine.get_telemetry()
            facts = engine._store.query_facts(limit=10000)
            n_facts = len(facts)
            logger.info("VC [%s]: running supersession over %d facts...", conv.conv_id, n_facts)
            t0 = time.time()
            engine.run_supersession()
            elapsed = time.time() - t0
            telem_after = engine.get_telemetry()
            n_superseded = telem_after.total().call_count - telem.total().call_count
            logger.info("VC [%s]: supersession done in %.1fs", conv.conv_id, elapsed)

        # Dedup facts (once after all compaction is done)
        from virtual_context.ingest.supersession import dedup_facts
        deduped = dedup_facts(engine._store)
        if deduped:
            logger.info("VC [%s]: deduped %d exact-duplicate facts", conv.conv_id, deduped)

        # Save state
        engine._save_state(messages)

        # Persist telemetry
        telem = engine.get_telemetry()
        telem_total = telem.total()
        if telem_total.call_count > 0:
            telemetry_path = conv_cache / "telemetry.json"
            telemetry_path.write_text(json.dumps(telem.to_dict(), indent=2))
            logger.info("VC [%s]: telemetry saved (%d calls, $%.4f)",
                         conv.conv_id, telem_total.call_count, telem_total.cost_usd)

    return engine, messages, stats


def query_question(
    engine: VirtualContextEngine,
    messages: list[Message],
    question: LoCoMoQuestion,
    conv: LoCoMoConversation,
    budget: BudgetTracker,
    *,
    reader_model: str = "gemini-3-pro-preview",
    reader_provider: str = "gemini",
    reader_auth_mode: str = "auto",
    cache_dir: Path | None = None,
    curation_enabled: bool = False,
    explain: bool = False,
) -> dict:
    """Query the VC engine for a single LocOMo question."""
    base_cache = cache_dir or DEFAULT_CACHE_DIR
    conv_cache = base_cache / conv.conv_id

    # Resolve API key
    if reader_provider in {"openai", "openai-codex"}:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        key_env = API_KEY_ENVS.get(reader_provider, "ANTHROPIC_API_KEY")
        api_key = os.environ.get(key_env, "")

    timings: dict[str, float] = {}

    # Retrieve context
    t0 = time.time()
    question_prompt = _format_question_prompt(question, conv, explain=explain)
    assembled = engine.on_message_inbound(question_prompt, messages)
    timings["retrieve_s"] = round(time.time() - t0, 1)

    tags_matched = assembled.matched_tags if assembled else []
    tokens_injected = assembled.total_tokens if assembled else 0

    # Determine if compacted
    compacted = engine._engine_state.compacted_through > 0 and assembled and assembled.prepend_text

    # Build prompt
    if compacted:
        system_prompt = assembled.prepend_text
        user_prompt = question_prompt
    else:
        system_prompt = _format_full_haystack(conv)
        user_prompt = question_prompt

    budget_breakdown = {}
    assembled_total_tokens = 0
    assembled_budget_breakdown = {}
    if assembled:
        assembled_total_tokens = assembled.total_tokens
        if hasattr(assembled, "budget_breakdown"):
            assembled_budget_breakdown = assembled.budget_breakdown or {}

    # Reader query
    t0 = time.time()
    reader_api_url = None
    if reader_provider == "anthropic":
        reader_api_url = "https://api.anthropic.com/v1/messages"
    elif reader_provider == "openai":
        reader_api_url = "https://api.openai.com/v1/chat/completions"
    elif reader_provider == "openrouter":
        reader_api_url = "https://openrouter.ai/api/v1/chat/completions"

    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_prompt}],
        model=reader_model,
        system=system_prompt,
        max_tokens=4096,
        api_key=api_key,
        api_url=reader_api_url,
        temperature=0.0,
        force_tools=True,
        require_tools=bool(compacted),
        provider=reader_provider,
    )
    timings["query_s"] = round(time.time() - t0, 1)

    hypothesis = (loop_result.text or "").strip()
    reader_input = loop_result.input_tokens
    reader_output = loop_result.output_tokens

    tool_calls_log = []
    for tc in (loop_result.tool_calls or []):
        tool_calls_log.append({
            "tool": tc.tool_name,
            "input": tc.tool_input,
            "result": tc.result_json[:200] if tc.result_json else "",
            "duration_ms": round(tc.duration_ms, 1) if tc.duration_ms else 0,
        })

    # Dump payload
    _dump_payload_log(
        question_id=question.question_id,
        cache_dir=conv_cache,
        question_text=question.question,
        gold_answer=question.answer,
        category=question.category,
        tags_matched=tags_matched,
        tokens_injected=tokens_injected,
        budget_breakdown=budget_breakdown,
        assembled_total_tokens=assembled_total_tokens,
        assembled_budget_breakdown=assembled_budget_breakdown,
        tool_calls=tool_calls_log,
        continuation_count=loop_result.continuation_count,
        stop_reason=loop_result.stop_reason or "",
        raw_requests=loop_result.raw_requests or [],
        raw_responses=loop_result.raw_responses or [],
        hypothesis=hypothesis,
        reader_input=reader_input,
        reader_output=reader_output,
        timings=timings,
        cached=engine._engine_state.compacted_through > 0,
    )

    return {
        "hypothesis": hypothesis,
        "input_tokens": reader_input,
        "output_tokens": reader_output,
        "tags_matched": tags_matched,
        "tokens_injected": tokens_injected,
        "tool_calls": tool_calls_log,
        "continuation_count": loop_result.continuation_count,
        "stop_reason": loop_result.stop_reason or "",
        "timings": timings,
        "cached": engine._engine_state.compacted_through > 0,
    }
