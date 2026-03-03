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
    "openai-codex": "https://chatgpt.com/backend-api/codex/responses",
    # Gemini URL is model-dependent, handled by adapter
}
API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR = DEFAULT_CACHE_DIR  # kept for backward compat (clear_cache uses it)

OPENAI_OAUTH_TOKEN_ENVS: tuple[str, ...] = (
    "OPENAI_OAUTH_ACCESS_TOKEN",
    "OPENAI_OAUTH_TOKEN",
    "OPENAI_ACCESS_TOKEN",
)


def _cache_dir_for(question_id: str, cache_dir: Path | None = None) -> Path:
    """Stable cache directory for a question's ingested + compacted state."""
    return (cache_dir or DEFAULT_CACHE_DIR) / question_id


def _load_openai_oauth_token_from_file(path: Path) -> str:
    """Best-effort OAuth token loader for Codex/OpenAI auth JSON files."""
    if not path.exists():
        return ""

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    data = raw

    # Tolerate a few likely shapes; return first non-empty string token.
    candidates = [
        data.get("access_token"),
        data.get("token"),
        data.get("openai_access_token"),
        (data.get("tokens") or {}).get("access_token") if isinstance(data.get("tokens"), dict) else "",
        (data.get("openai") or {}).get("access_token") if isinstance(data.get("openai"), dict) else "",
        (data.get("oauth") or {}).get("access_token") if isinstance(data.get("oauth"), dict) else "",
    ]
    for token in candidates:
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


def _resolve_openai_bearer_token(auth_mode: str = "auto") -> str:
    """Resolve OpenAI bearer token via OAuth and/or API key based on mode.

    Returns a token suitable for the Authorization Bearer header.
    """
    mode = (auth_mode or "auto").strip().lower()

    if mode in {"auto", "oauth"}:
        for env_name in OPENAI_OAUTH_TOKEN_ENVS:
            tok = os.environ.get(env_name, "").strip()
            if tok:
                logger.info("OpenAI auth: using OAuth token from %s", env_name)
                return tok

        explicit_file = os.environ.get("OPENAI_OAUTH_TOKEN_FILE", "").strip()
        search_paths: list[Path] = []
        if explicit_file:
            search_paths.append(Path(explicit_file).expanduser())
        codex_home = os.environ.get("CODEX_HOME", "").strip()
        if codex_home:
            search_paths.append(Path(codex_home).expanduser() / "auth.json")
        search_paths.append(Path.home() / ".codex" / "auth.json")

        for path in search_paths:
            tok = _load_openai_oauth_token_from_file(path)
            if tok:
                logger.info("OpenAI auth: using OAuth token from %s", path)
                return tok

    if mode in {"auto", "api-key"}:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            logger.info("OpenAI auth: using API key from OPENAI_API_KEY")
            return key

    return ""


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
    tagger_provider: str = "anthropic",
    tagger_model: str = "claude-haiku-4-5-20251001",
    summarizer_provider: str | None = None,
    summarizer_model: str | None = None,
    openai_bearer_token: str = "",
    tagger_mode: str = "combined",
    fact_provider: str | None = None,
    fact_model: str | None = None,
) -> dict:
    """Build a VC config dict for benchmark use."""
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
            "api_key": openai_bearer_token,
        }
    if tagger_provider == "ollama_native" or chosen_summarizer_provider == "ollama_native":
        providers["ollama_native"] = {
            "type": "ollama_native",
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "model": tagger_model if tagger_provider == "ollama_native" else chosen_summarizer_model,
            "num_predict": 500,
            "force_json": True,
        }
    if tagger_provider == "openrouter" or chosen_summarizer_provider == "openrouter":
        providers["openrouter"] = {
            "type": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": tagger_model if tagger_provider == "openrouter" else chosen_summarizer_model,
            "api_key_env": "OPENROUTER_API_KEY",
        }
    # Fact provider for split mode (if different from tagger)
    if fact_provider and fact_provider not in providers:
        if fact_provider == "ollama_native":
            providers[fact_provider] = {
                "type": "ollama_native",
                "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                "model": fact_model or tagger_model,
                "num_predict": 500,
                "force_json": True,
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
            "context_bleed_threshold": 0,  # disable embedding gate for batch processing
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
            # Match proxy behavior: inbound query tagging should use embedding
            # vocabulary matching (no LLM call) for retrieval selection.
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
                "anthropic": {
                    "input_per_1k": 0.00025,
                    "output_per_1k": 0.00125,
                },
                "openai": {
                    "input_per_1k": 0.00025,
                    "output_per_1k": 0.00200,
                },
            },
        },
    }

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
            # Actual prompt payload sent to the reader model (estimate).
            "tokens_injected": tokens_injected,
            # Backward-compatible key: now represents actual prompt-token breakdown.
            "budget_breakdown": budget_breakdown,
            # Assembly accounting kept separately for debugging/diagnostics.
            "assembled_total_tokens": assembled_total_tokens,
            "assembled_budget_breakdown": assembled_budget_breakdown,
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


def _format_full_haystack_prompt(
    q: LongMemEvalQuestion,
    *,
    diagnostic_rationale: bool = False,
) -> str:
    """Format full history using the baseline LongMemEval prompt template."""
    parts = [
        "I will give you several history chats between you and a user.",
        "Please answer the question based on the relevant chat history.",
        "",
        "History Chats:",
    ]

    for i, (session, date) in enumerate(zip(q.haystack_sessions, q.haystack_dates), 1):
        parts.append("")
        parts.append(f"### Session {i}:")
        parts.append(f"Session Date: {date}")
        parts.append("Session Content:")
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")

    parts.append("")
    parts.append(_question_block(q, diagnostic_rationale=diagnostic_rationale))
    return "\n".join(parts)


def _question_block(
    q: LongMemEvalQuestion,
    *,
    diagnostic_rationale: bool = False,
) -> str:
    """Build the question tail block, optionally requiring diagnostic rationale."""
    parts = [
        f"Current Date: {q.question_date}",
        f"Question: {q.question}",
    ]
    if diagnostic_rationale:
        parts.extend([
            "Output format (required):",
            "FINAL_ANSWER: <single-sentence direct answer>",
            "EVIDENCE_1: <short quote snippet> | session=<session date if available>",
            "EVIDENCE_2: <optional second quote> | session=<session date if available>",
            "CONFLICT_HANDLING: If evidence comes from different sessions, the most recent session ALWAYS defines the current answer (even if it uses future-intent language like 'planning to' or 'looking forward to'). State which session you used.",
            "Use the format exactly and keep each line concise.",
        ])
    parts.append("Answer:")
    return "\n".join(parts)


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
    reader_auth_mode: str = "auto",
    tagger_provider: str = "anthropic",
    tagger_model: str = "claude-haiku-4-5-20251001",
    summarizer_provider: str | None = None,
    internal_openai_auth_mode: str = "auto",
    reader_diagnostic_rationale: bool = False,
    tagger_mode: str = "combined",
    fact_provider: str | None = None,
    fact_model: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Run the VC pipeline for a single question.

    Steps: convert sessions → ingest_history → compact_manual loop → on_message_inbound → query reader

    When a cached ingestion+compaction exists for this question_id, steps 2-4
    are skipped entirely (~10 min saved). Use fresh=True to force re-ingestion.

    Parameters
    ----------
    reader_provider : str
        LLM provider for the reader model: ``"anthropic"``, ``"openai"``,
        ``"openai-codex"``, or ``"gemini"``.

    Returns dict with: hypothesis, input_tokens, output_tokens, cost,
                       tags_matched, tokens_injected, compaction_events, timings.
    """
    # Resolve API key: explicit > env var for provider > ANTHROPIC_API_KEY fallback
    key_env = API_KEY_ENVS.get(reader_provider, "ANTHROPIC_API_KEY")
    if api_key:
        pass
    elif reader_provider in {"openai", "openai-codex"}:
        api_key = _resolve_openai_bearer_token(reader_auth_mode)
    else:
        api_key = os.environ.get(key_env, "")
    # Resolve internal provider credentials used by ingestion/compaction.
    chosen_summarizer_provider = summarizer_provider or tagger_provider
    needs_anthropic = (
        tagger_provider == "anthropic"
        or chosen_summarizer_provider == "anthropic"
    )
    needs_openai = (
        tagger_provider == "openai"
        or chosen_summarizer_provider == "openai"
    )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    internal_openai_token = ""
    if needs_openai:
        internal_openai_token = _resolve_openai_bearer_token(internal_openai_auth_mode)
    if not api_key:
        if reader_provider in {"openai", "openai-codex"}:
            if reader_auth_mode == "oauth":
                raise ValueError(
                    "OpenAI OAuth token not found. Set one of "
                    "OPENAI_OAUTH_ACCESS_TOKEN / OPENAI_OAUTH_TOKEN / OPENAI_ACCESS_TOKEN, "
                    "or point OPENAI_OAUTH_TOKEN_FILE to an auth.json file."
                )
            raise ValueError(
                "OpenAI auth not found. Set OPENAI_API_KEY, or use OAuth by setting one of "
                "OPENAI_OAUTH_ACCESS_TOKEN / OPENAI_OAUTH_TOKEN / OPENAI_ACCESS_TOKEN."
            )
        raise ValueError(f"{key_env} not set")
    if needs_anthropic and not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set (needed for tagger/summarizer)")
    if needs_openai and not internal_openai_token:
        raise ValueError(
            "OpenAI auth not found for tagger/summarizer. Set OPENAI_API_KEY or "
            "an OAuth token (OPENAI_OAUTH_ACCESS_TOKEN / OPENAI_OAUTH_TOKEN / OPENAI_ACCESS_TOKEN)."
        )

    timings: dict[str, float] = {}

    # 1. Build engine — use stable cache dir for resumable state
    q_cache_dir = _cache_dir_for(question.question_id, cache_dir)
    if fresh and q_cache_dir.exists():
        shutil.rmtree(q_cache_dir)
        logger.info("VC [%s]: cleared cache (--fresh)", question.question_id)
    q_cache_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = str(q_cache_dir)

    if recompact and not fresh:
        _clear_compaction_state(storage_dir, question.question_id)

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"bench-{question.question_id}",
        tagger_provider=tagger_provider,
        tagger_model=tagger_model,
        summarizer_provider=summarizer_provider,
        summarizer_model=summarizer_model,
        openai_bearer_token=internal_openai_token,
        tagger_mode=tagger_mode,
        fact_provider=fact_provider,
        fact_model=fact_model,
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

    # Persist/restore pre-reader LLM cost data so cached reruns don't lose it.
    # Store the actual provider so we don't misattribute Ollama/Qwen tokens as Anthropic.
    cost_snapshot_path = q_cache_dir / "cost_snapshot.json"
    pre_reader_provider = tagger_provider  # provider actually used for tagging/compaction
    if haiku_input > 0 or haiku_output > 0:
        # Fresh or recompact run — save costs with actual provider
        cost_snapshot_path.write_text(json.dumps({
            "haiku_input_tokens": haiku_input,
            "haiku_output_tokens": haiku_output,
            "haiku_calls": haiku_calls,
            "provider": pre_reader_provider,
        }))
    elif cost_snapshot_path.exists():
        # Cache hit — restore costs from snapshot
        snap = json.loads(cost_snapshot_path.read_text())
        haiku_input = snap.get("haiku_input_tokens", 0)
        haiku_output = snap.get("haiku_output_tokens", 0)
        haiku_calls = snap.get("haiku_calls", 0)
        pre_reader_provider = snap.get("provider", "anthropic")  # default for old snapshots
        logger.info("VC [%s]: restored pre-reader costs from snapshot: %d in / %d out (%d calls, provider=%s)",
                    question.question_id, haiku_input, haiku_output, haiku_calls, pre_reader_provider)

    # 4. Retrieve context for the question
    t0 = time.time()
    question_prompt = f"Current Date: {question.question_date}\nQuestion: {question.question}"
    assembled = engine.on_message_inbound(question_prompt, messages)
    timings["retrieve_s"] = round(time.time() - t0, 1)

    tags_matched = assembled.matched_tags
    prepend_text = assembled.prepend_text
    assembled_total_tokens = assembled.total_tokens
    assembled_budget_breakdown = assembled.budget_breakdown

    prepend_tokens = len(prepend_text) // 4 if prepend_text else 0
    logger.info(
        "VC [%s]: assembled total=%d tokens, prepend_text=%d chars (~%d tokens), tags: %s",
        question.question_id, assembled_total_tokens, len(prepend_text), prepend_tokens, tags_matched[:10],
    )
    logger.info(
        "VC [%s]: assembled_budget_breakdown: %s",
        question.question_id, assembled_budget_breakdown,
    )
    if verbose and prepend_text:
        logger.info("VC [%s]: prepend_text preview:\n%s", question.question_id, prepend_text[:1000])

    # Record pre-reader LLM cost (tagging + compaction) — only for paid providers.
    # Ollama/Qwen has no per-token API cost so skip budget recording for it.
    if haiku_input > 0 or haiku_output > 0:
        if pre_reader_provider == "anthropic":
            budget.record(
                label=f"vc_internal:{question.question_id}",
                model="claude-haiku-4-5-20251001",
                input_tokens=haiku_input,
                output_tokens=haiku_output,
            )
        logger.info(
            "VC [%s]: pre-reader actuals: %d in / %d out (%d calls, provider=%s)",
            question.question_id, haiku_input, haiku_output, haiku_calls, pre_reader_provider,
        )

    # 5. Send retrieved context + question to reader model (with tool loop)
    t0 = time.time()

    # Mirror proxy behavior: VC context goes into system/instructions,
    # user message contains ONLY the question.
    context_hint = assembled.context_hint
    use_raw_history = engine._compacted_through == 0
    question_tail = _question_block(
        question, diagnostic_rationale=reader_diagnostic_rationale,
    )
    if use_raw_history:
        # Pre-compaction: full raw history in system, question in user
        system_prompt = _format_full_haystack_prompt(
            question, diagnostic_rationale=reader_diagnostic_rationale,
        )
        user_prompt = question_tail
    else:
        # Post-compaction: VC summaries + context hint in system, question in user
        vc_summaries = "\n\n".join(assembled.tag_sections.values())
        system_parts = []
        if context_hint:
            system_parts.append(context_hint)
        system_parts.append(
            "I will give you several history chats between you and a user. "
            "Please answer the question based on the relevant chat history.\n\n\n"
            "History Chats:\n\n"
            f"{vc_summaries}"
        )
        # Include protected (uncompacted) conversation messages if present.
        # The assembler trims these to fit the token budget, but the runner
        # was previously ignoring them — meaning the protected-zone tail of
        # each session was lost.
        if assembled.facts_text:
            system_parts.append(assembled.facts_text)
        if assembled.conversation_history:
            conv_lines = [
                f"{msg.role.capitalize()}: {msg.content}"
                for msg in assembled.conversation_history
            ]
            system_parts.append(
                "Recent Conversation:\n\n" + "\n\n".join(conv_lines)
            )
        system_prompt = "\n\n".join(system_parts)
        user_prompt = question_tail
    # Report actual payload size sent to the reader model (not assembler internals).
    system_tokens_est = len(system_prompt) // 4 if system_prompt else 0
    user_tokens_est = len(user_prompt) // 4 if user_prompt else 0
    tokens_injected = system_tokens_est + user_tokens_est
    prompt_budget_breakdown = {
        "system": system_tokens_est,
        "user": user_tokens_est,
        "total": tokens_injected,
    }
    logger.info(
        "VC [%s]: reader prompt tokens est=%d (system=%d, user=%d)",
        question.question_id, tokens_injected, system_tokens_est, user_tokens_est,
    )
    logger.info(
        "VC [%s]: reader prompt mode=%s",
        question.question_id,
        "raw-history" if use_raw_history else "summary-history",
    )

    reader_api_url = API_URLS.get(reader_provider, "")
    require_tools = engine._compacted_through > 0
    logger.info(
        "VC [%s]: reader tool policy: %s (compacted_through=%d)",
        question.question_id,
        "required" if require_tools else "optional",
        engine._compacted_through,
    )
    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_prompt}],
        model=reader_model,
        system=system_prompt,
        max_tokens=1024,
        api_key=api_key,
        api_url=reader_api_url,
        temperature=0.0,
        force_tools=True,
        require_tools=require_tools,
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
        cache_dir=q_cache_dir,
        question_text=question.question,
        gold_answer=question.answer,
        question_date=question.question_date,
        tags_matched=tags_matched,
        tokens_injected=tokens_injected,
        budget_breakdown=prompt_budget_breakdown,
        assembled_total_tokens=assembled_total_tokens,
        assembled_budget_breakdown=assembled_budget_breakdown,
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
        "assembled_total_tokens": assembled_total_tokens,
        "assembled_budget_breakdown": assembled_budget_breakdown,
        "prompt_token_breakdown": prompt_budget_breakdown,
        "tokens_injected": tokens_injected,
        "compaction_events": compaction_events,
        "turns_ingested": turns_ingested,
        "timings": timings,
        "cached": cached,
        "tool_calls": tool_calls_log,
        "continuation_count": loop_result.continuation_count,
        "stop_reason": loop_result.stop_reason,
    }


def run_vc_ingest_only(
    question: LongMemEvalQuestion,
    context_window: int = 65536,
    fresh: bool = False,
    recompact: bool = False,
    tagger_provider: str = "anthropic",
    tagger_model: str = "claude-haiku-4-5-20251001",
    summarizer_provider: str | None = None,
    summarizer_model: str | None = None,
    tagger_mode: str = "combined",
    fact_provider: str | None = None,
    fact_model: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Run ingest + compact only, skip reader and judge. Returns stats dict."""
    timings: dict[str, float] = {}

    cache_dir = _cache_dir_for(question.question_id, cache_dir)
    if fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = str(cache_dir)

    if recompact and not fresh:
        _clear_compaction_state(storage_dir, question.question_id)

    # Resolve internal provider credentials
    chosen_summarizer_provider = summarizer_provider or tagger_provider
    needs_anthropic = tagger_provider == "anthropic" or chosen_summarizer_provider == "anthropic"
    needs_openai = tagger_provider == "openai" or chosen_summarizer_provider == "openai"
    openai_bearer_token = ""
    if needs_openai:
        openai_bearer_token = _resolve_openai_bearer_token("auto")

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"bench-{question.question_id}",
        tagger_provider=tagger_provider,
        tagger_model=tagger_model,
        summarizer_provider=summarizer_provider,
        summarizer_model=summarizer_model,
        openai_bearer_token=openai_bearer_token,
        tagger_mode=tagger_mode,
        fact_provider=fact_provider,
        fact_model=fact_model,
    )
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    messages = _sessions_to_messages(question)

    # Check cache status
    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._compacted_through > 0

    if fully_cached:
        logger.info("VC [%s]: CACHE HIT — skipping ingest+compact", question.question_id)
        return {
            "turns_ingested": n_index_entries,
            "compaction_events": 0,
            "cached": True,
            "timings": {"ingest_s": 0.0, "compact_s": 0.0},
        }

    # Ingest
    tags_only = not fully_cached and n_index_entries > 0
    if tags_only:
        timings["ingest_s"] = 0.0
        turns_ingested = n_index_entries
    else:
        t0 = time.time()
        _ingest_start = time.time()

        def _progress(done: int, total: int, entry: object) -> None:
            if done % 20 == 0 or done == total:
                elapsed = time.time() - _ingest_start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info("  Ingest [%s]: %d/%d (%.1f/s)", question.question_id, done, total, rate)

        turns_ingested = engine.ingest_history(messages, progress_callback=_progress)
        timings["ingest_s"] = round(time.time() - t0, 1)

    # Compact
    t0 = time.time()
    compaction_events = 0
    while True:
        report = engine.compact_manual(messages)
        if report is None:
            break
        compaction_events += 1
    timings["compact_s"] = round(time.time() - t0, 1)

    engine._save_state(messages)
    engine.close()

    return {
        "turns_ingested": turns_ingested,
        "compaction_events": compaction_events,
        "cached": False,
        "timings": timings,
    }
