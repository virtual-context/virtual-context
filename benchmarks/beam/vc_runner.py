"""VC pipeline for BEAM: ingest conversation once, query per probing question.

API keys: benchmarks/beam/.env (gitignored). Source it before running.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

from benchmarks.longmemeval.cost import BudgetTracker
from .dataset import BEAMConversation, BEAMQuestion, flatten_messages

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def _chat_to_messages(conv: BEAMConversation) -> list[Message]:
    """Convert BEAM chat data into flat Message list for VC ingestion.

    Embeds time_anchor metadata when present. Merges consecutive same-role
    messages. Pads to even count (VC needs user/assistant pairs).
    """
    flat = flatten_messages(conv.raw_data, conv.chat_size)
    messages: list[Message] = []

    for i, m in enumerate(flat):
        role = m["role"]
        content = m["content"]
        time_anchor = m.get("time_anchor", "")

        # Embed time anchor on first message of each new time period
        if time_anchor and (i == 0 or flat[i - 1].get("time_anchor", "") != time_anchor):
            content = f"[Session from {time_anchor}] {content}"

        # Merge consecutive same-role messages
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


# ---------------------------------------------------------------------------
# VC config
# ---------------------------------------------------------------------------

def _build_vc_config(
    *,
    context_window: int = 65536,
    storage_dir: str | None = None,
    session_id: str = "",
    tagger_provider: str = "openrouter",
    tagger_model: str,
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
    """Build a VC config dict for BEAM benchmark use."""
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
            "max_summary_tokens": 1200,
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


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _clear_compaction_state(cache_dir: str, conv_id: str) -> None:
    """Clear compacted segments and tag summaries but keep TurnTagIndex."""
    import sqlite3
    db_path = Path(cache_dir) / "store.db"
    if not db_path.exists():
        logger.warning("No DB found at %s -- nothing to clear", db_path)
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


# ---------------------------------------------------------------------------
# Payload logging
# ---------------------------------------------------------------------------

def _dump_payload_log(
    *,
    question_id: str,
    cache_dir: Path,
    question_text: str,
    gold_answer: str,
    category: str,
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
    """Dump payload log for post-run analysis (single file, matches LocOMo pattern)."""
    http_conversation = []
    for i in range(max(len(raw_requests), len(raw_responses))):
        if i < len(raw_requests):
            http_conversation.append({"step": i + 1, "direction": "REQUEST", "body": raw_requests[i]})
        if i < len(raw_responses):
            http_conversation.append({"step": i + 1, "direction": "RESPONSE", "body": raw_responses[i]})

    payload = {
        "_description": "Raw HTTP payloads for BEAM question.",
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

    # Chain analytics (matches LocOMo pattern)
    try:
        from benchmarks.longmemeval.chain_analysis import analyze_tool_chain
        payload["chain_analysis"] = analyze_tool_chain(tool_calls)
    except ImportError:
        pass

    payloads_dir = cache_dir / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = payloads_dir / f"{question_id}_{ts}.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as e:
        logger.warning("Failed to save payload log for %s: %s", question_id, e)


# ---------------------------------------------------------------------------
# Ingestion (per-conversation, cached)
# ---------------------------------------------------------------------------

def ingest_conversation(
    conv: BEAMConversation,
    *,
    context_window: int = 65536,
    clear_cache: bool = False,
    recompact: bool = False,
    require_fully_cached: bool = False,
    tagger_provider: str = "openrouter",
    tagger_model: str,
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
    """Ingest a BEAM conversation into VC. Returns (engine, messages, stats).

    If cache exists, returns cached engine without re-ingesting.
    Cache is stored at: cache/{chat_size}/{conv_id}/
    """
    base_cache = cache_dir or DEFAULT_CACHE_DIR
    conv_cache = base_cache / conv.chat_size / conv.conv_id
    if clear_cache and conv_cache.exists():
        shutil.rmtree(conv_cache)
        logger.info("VC [%s]: deleted cache directory (--clear-cache)", conv.conv_id)
    conv_cache.mkdir(parents=True, exist_ok=True)
    storage_dir = str(conv_cache)

    if recompact and not clear_cache:
        _clear_compaction_state(storage_dir, conv.conv_id)

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"beam-{conv.conv_id}",
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

    # Set reference_date from conversation timeline (matches LongMemEval pattern)
    # so temporal resolution uses conversation dates, not today's date.
    messages = _chat_to_messages(conv)
    _last_anchor = _extract_last_time_anchor(conv)
    if _last_anchor:
        engine.reference_date = _last_anchor
        logger.info("VC [%s]: reference_date set to %s", conv.conv_id, engine.reference_date)
    n_pairs = len(messages) // 2

    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._engine_state.compacted_through > 0
    tags_only = not fully_cached and n_index_entries > 0

    if require_fully_cached and not fully_cached:
        raise RuntimeError(
            "Cache-only mode requires a fully compacted cache, but this conversation "
            f"has compacted_through={engine._engine_state.compacted_through} and "
            f"{n_index_entries}/{n_pairs} indexed turns. Refusing to ingest or compact."
        )

    stats: dict = {"conv_id": conv.conv_id, "turns_ingested": 0, "compaction_events": 0}

    # Shared progress callback for ingest / resume
    _ingest_start = time.time()

    def _progress(done: int, total: int, entry: object) -> None:
        if done % 20 == 0 or done == total:
            elapsed = time.time() - _ingest_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            logger.info("  Ingest: %d/%d (%.1f turns/s, ETA %.0fs)", done, total, rate, eta)

    if fully_cached:
        logger.info(
            "VC [%s]: CACHE HIT -- %d turns indexed, compacted_through=%d",
            conv.conv_id, n_index_entries, engine._engine_state.compacted_through,
        )
        stats["turns_ingested"] = n_index_entries
        stats["compaction_events"] = -1
        stats["cached"] = True
    elif tags_only:
        # Resume partial ingestion if we didn't finish tagging all pairs last time.
        highest_indexed = max(
            (e.turn_number for e in engine._turn_tag_index.entries), default=-1
        )
        resume_from = highest_indexed + 1
        if resume_from < n_pairs:
            remaining = n_pairs - resume_from
            logger.info(
                "VC [%s]: RESUME -- %d/%d turns indexed, ingesting remaining %d from turn %d",
                conv.conv_id, n_index_entries, n_pairs, remaining, resume_from,
            )
            stats["cached"] = False
            t0 = time.time()
            remaining_msgs = messages[2 * resume_from:]
            new_turns = engine.ingest_history(
                remaining_msgs,
                progress_callback=_progress,
                turn_offset=resume_from,
            )
            stats["turns_ingested"] = n_index_entries + new_turns
            logger.info(
                "VC [%s]: resumed %d turns in %.1fs (total indexed: %d)",
                conv.conv_id, new_turns, time.time() - t0, n_index_entries + new_turns,
            )
        else:
            logger.info(
                "VC [%s]: RECOMPACT -- %d turns indexed, re-running compaction",
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

        # Supersession (separate pass, after compaction — matches LocOMo pattern)
        if supersession:
            facts = engine._store.query_facts(limit=10000)
            n_facts = len(facts)
            logger.info("VC [%s]: running supersession over %d facts...", conv.conv_id, n_facts)
            t0_ss = time.time()
            engine.run_supersession()
            logger.info("VC [%s]: supersession done in %.1fs", conv.conv_id, time.time() - t0_ss)

        # Dedup facts
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


# ---------------------------------------------------------------------------
# Query (per-question)
# ---------------------------------------------------------------------------

def _extract_last_time_anchor(conv: BEAMConversation) -> "date | None":
    """Extract the last time_anchor from the conversation as a date object."""
    from datetime import date as _date
    flat = flatten_messages(conv.raw_data, conv.chat_size)
    last_anchor = ""
    for m in flat:
        anchor = m.get("time_anchor", "")
        if anchor:
            last_anchor = anchor
    if not last_anchor:
        return None
    # BEAM time_anchors are like "July-01-2024", "March-01-2025"
    import re
    # Try Month-DD-YYYY
    m = re.match(r"(\w+)-(\d{1,2})-(\d{4})", last_anchor)
    if m:
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(last_anchor, "%B-%d-%Y")
            return dt.date()
        except ValueError:
            pass
    # Try YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", last_anchor)
    if m:
        try:
            return _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _format_full_haystack(conv: BEAMConversation, messages: list[Message]) -> str:
    """Pre-compaction fallback: full raw history as system prompt."""
    parts = ["Below is a multi-turn conversation. Use it to answer the question that follows.\n"]
    for m in messages:
        parts.append(f"{m.role.upper()}: {m.content}")
    return "\n".join(parts)


def _format_question_prompt(question: BEAMQuestion) -> str:
    """Build question prompt for the reader model."""
    return (
        "Based on the conversation history, answer the following question. "
        "Only provide the answer without any explanations.\n\n"
        f"Question: {question.question}"
    )


def query_question(
    engine: VirtualContextEngine,
    messages: list[Message],
    question: BEAMQuestion,
    conv: BEAMConversation,
    budget: BudgetTracker,
    *,
    reader_model: str = "claude-sonnet-4-20250514",
    reader_provider: str = "anthropic",
    cache_dir: Path | None = None,
) -> dict:
    """Query the VC engine for a single BEAM probing question."""
    base_cache = cache_dir or DEFAULT_CACHE_DIR
    conv_cache = base_cache / conv.chat_size / conv.conv_id

    # --- Per-question engine log capture ---
    payloads_dir = conv_cache / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    engine_log_path = payloads_dir / f"{question.question_id}_{log_ts}.engine.log"
    _engine_fh = logging.FileHandler(engine_log_path, mode="w")
    _engine_fh.setLevel(logging.DEBUG)
    _engine_fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    _root_logger = logging.getLogger()
    _root_logger.addHandler(_engine_fh)
    _prev_root_level = _root_logger.level
    _root_logger.setLevel(logging.DEBUG)

    # Resolve API key
    key_env = API_KEY_ENVS.get(reader_provider, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(key_env, "")

    timings: dict[str, float] = {}
    cached = engine._engine_state.compacted_through > 0

    # --- Haiku cost tracking (matches LongMemEval pattern) ---
    pre_reader_telem = engine.get_telemetry().total()
    haiku_input = pre_reader_telem.input_tokens
    haiku_output = pre_reader_telem.output_tokens
    haiku_calls = pre_reader_telem.call_count

    cost_snapshot_path = conv_cache / "cost_snapshot.json"
    if haiku_input > 0 or haiku_output > 0:
        cost_snapshot_path.write_text(json.dumps({
            "haiku_input_tokens": haiku_input,
            "haiku_output_tokens": haiku_output,
            "haiku_calls": haiku_calls,
        }))
    elif cost_snapshot_path.exists():
        snap = json.loads(cost_snapshot_path.read_text())
        haiku_input = snap.get("haiku_input_tokens", 0)
        haiku_output = snap.get("haiku_output_tokens", 0)
        haiku_calls = snap.get("haiku_calls", 0)
        logger.info("VC [%s]: restored pre-reader costs from snapshot: %d in / %d out (%d calls)",
                    question.question_id, haiku_input, haiku_output, haiku_calls)

    # --- Retrieve context ---
    t0 = time.time()
    question_prompt = _format_question_prompt(question)
    assembled = engine.on_message_inbound(question_prompt, messages)
    timings["retrieve_s"] = round(time.time() - t0, 1)

    tags_matched = assembled.matched_tags if assembled else []
    assembled_total_tokens = assembled.total_tokens if assembled else 0
    assembled_budget_breakdown = (assembled.budget_breakdown or {}) if assembled and hasattr(assembled, "budget_breakdown") else {}

    # --- Build reader prompt ---
    use_raw_history = engine._engine_state.compacted_through == 0
    if use_raw_history:
        system_prompt = _format_full_haystack(conv, messages)
        vc_injection = ""
    else:
        # VC content placed near last turn (matching proxy), not system prompt
        vc_parts = []
        if assembled and assembled.context_hint:
            vc_parts.append(assembled.context_hint)
        vc_summaries = "\n\n".join(assembled.tag_sections.values()) if assembled else ""
        if vc_summaries:
            vc_parts.append(
                "Below is a multi-turn conversation history, organized by topic. "
                "Use it to answer the question that follows.\n\n"
                f"Conversation History:\n\n{vc_summaries}"
            )
        if assembled and assembled.facts_text:
            vc_parts.append(assembled.facts_text)
        if assembled and assembled.conversation_history:
            conv_lines = [
                f"{msg.role.capitalize()}: {msg.content}"
                for msg in assembled.conversation_history
            ]
            vc_parts.append(
                "Recent Conversation:\n\n" + "\n\n".join(conv_lines)
            )
        vc_injection = "\n\n".join(vc_parts)
        system_prompt = ""

    # Payload size estimate
    vc_tokens_est = len(vc_injection) // 4 if vc_injection else 0
    system_tokens_est = len(system_prompt) // 4 if system_prompt else 0
    user_tokens_est = len(question_prompt) // 4 if question_prompt else 0
    tokens_injected = system_tokens_est + user_tokens_est + vc_tokens_est
    prompt_budget_breakdown = {
        "system": system_tokens_est,
        "vc_injection": vc_tokens_est,
        "user": user_tokens_est,
        "total": tokens_injected,
    }

    # Record pre-reader LLM cost
    if haiku_input > 0 or haiku_output > 0:
        budget.record(
            label=f"beam-internal-{question.question_id}",
            model="claude-haiku-4-5-20251001",
            input_tokens=haiku_input,
            output_tokens=haiku_output,
        )

    # --- Reader query (matches LongMemEval pattern) ---
    t0 = time.time()
    reader_api_url = None
    if reader_provider == "anthropic":
        reader_api_url = "https://api.anthropic.com/v1/messages"
    elif reader_provider == "openai":
        reader_api_url = "https://api.openai.com/v1/chat/completions"
    elif reader_provider == "openrouter":
        reader_api_url = "https://openrouter.ai/api/v1/chat/completions"

    require_tools = engine._engine_state.compacted_through > 0

    # Place VC injection near last turn (matching proxy), not system prompt
    if vc_injection:
        user_content = [
            {"type": "text", "text": question_prompt},
            {"type": "text", "text": f"<system-reminder>\n{vc_injection}\n</system-reminder>"},
        ]
    else:
        user_content = question_prompt

    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_content}],
        model=reader_model,
        system=system_prompt,
        max_tokens=8192,
        api_key=api_key,
        api_url=reader_api_url,
        temperature=0.0,
        force_tools=True,
        require_tools=require_tools,
        provider=reader_provider,
    )
    timings["query_s"] = round(time.time() - t0, 1)

    hypothesis = (loop_result.text or "").strip()
    reader_input = loop_result.input_tokens
    reader_output = loop_result.output_tokens

    # Full tool result capture (matches LongMemEval -- no truncation)
    tool_calls_log = [
        {
            "tool": tc.tool_name,
            "input": tc.tool_input,
            "result": tc.result_json,
            "duration_ms": tc.duration_ms,
        }
        for tc in (loop_result.tool_calls or [])
    ]

    # Track reader costs
    budget.record(
        label=f"beam-reader-{question.question_id}",
        model=reader_model,
        input_tokens=reader_input,
        output_tokens=reader_output,
    )

    total_cost = sum(
        e.cost_usd for e in budget.entries
        if question.question_id in e.label and ("beam-internal" in e.label or "beam-reader" in e.label)
    )

    # Dump payload (single file + chain_analysis)
    _dump_payload_log(
        question_id=question.question_id,
        cache_dir=conv_cache,
        question_text=question.question,
        gold_answer=question.ideal_response,
        category=question.category,
        tags_matched=tags_matched,
        tokens_injected=tokens_injected,
        budget_breakdown=prompt_budget_breakdown,
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
        cached=cached,
    )

    # --- Tear down per-question engine log capture ---
    _root_logger.removeHandler(_engine_fh)
    _engine_fh.close()
    _root_logger.setLevel(_prev_root_level)

    # Tool chain summary (matches LongMemEval pattern)
    tool_chain = []
    for tc in tool_calls_log:
        inp = tc.get("input", {})
        if "query" in inp:
            detail = inp["query"]
        elif "tag" in inp:
            detail = inp["tag"]
        else:
            parts = [f"{k}={v}" for k, v in inp.items() if v]
            detail = ", ".join(parts) if parts else ""
        tool_chain.append(f"{tc['tool']}({detail})")

    return {
        "hypothesis": hypothesis,
        "input_tokens": reader_input,
        "output_tokens": reader_output,
        "haiku_input_tokens": haiku_input,
        "haiku_output_tokens": haiku_output,
        "haiku_calls": haiku_calls,
        "cost": round(total_cost, 6),
        "tags_matched": tags_matched,
        "assembled_total_tokens": assembled_total_tokens,
        "assembled_budget_breakdown": assembled_budget_breakdown,
        "prompt_token_breakdown": prompt_budget_breakdown,
        "tokens_injected": tokens_injected,
        "timings": timings,
        "cached": cached,
        "tool_chain": tool_chain,
        "tool_calls": tool_calls_log,
        "continuation_count": loop_result.continuation_count,
        "stop_reason": loop_result.stop_reason or "",
    }
