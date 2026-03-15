"""VC pipeline: ingest → compact → retrieve → query for a single MRCR question."""

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

from .dataset import MRCRQuestion

logger = logging.getLogger(__name__)

API_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "openai-codex": "https://chatgpt.com/backend-api/codex/responses",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR = DEFAULT_CACHE_DIR

OPENAI_OAUTH_TOKEN_ENVS: tuple[str, ...] = (
    "OPENAI_OAUTH_ACCESS_TOKEN",
    "OPENAI_OAUTH_TOKEN",
    "OPENAI_ACCESS_TOKEN",
)


def _cache_dir_for(question_id: str, cache_dir: Path | None = None) -> Path:
    """Stable cache directory for a question's ingested + compacted state."""
    return (cache_dir or DEFAULT_CACHE_DIR) / question_id


def _load_openai_oauth_token_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    candidates = [
        raw.get("access_token"),
        raw.get("token"),
        raw.get("openai_access_token"),
        (raw.get("tokens") or {}).get("access_token") if isinstance(raw.get("tokens"), dict) else "",
        (raw.get("openai") or {}).get("access_token") if isinstance(raw.get("openai"), dict) else "",
        (raw.get("oauth") or {}).get("access_token") if isinstance(raw.get("oauth"), dict) else "",
    ]
    for token in candidates:
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


def _resolve_openai_bearer_token(auth_mode: str = "auto") -> str:
    mode = (auth_mode or "auto").strip().lower()
    if mode in {"auto", "oauth"}:
        for env_name in OPENAI_OAUTH_TOKEN_ENVS:
            tok = os.environ.get(env_name, "").strip()
            if tok:
                return tok
        search_paths: list[Path] = []
        explicit_file = os.environ.get("OPENAI_OAUTH_TOKEN_FILE", "").strip()
        if explicit_file:
            search_paths.append(Path(explicit_file).expanduser())
        codex_home = os.environ.get("CODEX_HOME", "").strip()
        if codex_home:
            search_paths.append(Path(codex_home).expanduser() / "auth.json")
        search_paths.append(Path.home() / ".codex" / "auth.json")
        for path in search_paths:
            tok = _load_openai_oauth_token_from_file(path)
            if tok:
                return tok
    if mode in {"auto", "api-key"}:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            return key
    return ""


def clear_cache(question_ids: list[str] | None = None) -> int:
    """Remove cached VC state. Returns count of caches removed."""
    if not CACHE_DIR.exists():
        return 0
    if question_ids is None:
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
    curation_enabled: bool = False,
    curation_provider: str | None = None,
    curation_model: str | None = None,
    supersession: bool = False,
    supersession_provider: str | None = None,
    supersession_model: str | None = None,
) -> dict:
    """Build a VC config dict for MRCR benchmark use."""
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
        elif fact_provider == "openrouter":
            providers["openrouter"] = {
                "type": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": fact_model or tagger_model,
                "api_key_env": "OPENROUTER_API_KEY",
            }
    # Curation provider (if different from others already registered)
    chosen_curation_provider = curation_provider or chosen_summarizer_provider
    chosen_curation_model = curation_model or chosen_summarizer_model
    if chosen_curation_provider and chosen_curation_provider not in providers:
        if chosen_curation_provider == "openai":
            providers["openai"] = {
                "type": "generic_openai",
                "base_url": "https://api.openai.com/v1",
                "model": chosen_curation_model,
                "api_key": openai_bearer_token or os.environ.get("OPENAI_API_KEY", ""),
            }
        elif chosen_curation_provider == "openrouter":
            providers["openrouter"] = {
                "type": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": chosen_curation_model,
                "api_key_env": "OPENROUTER_API_KEY",
            }

    # Supersession provider: falls back to fact provider, then summarizer
    chosen_supersession_provider = supersession_provider or fact_provider or chosen_summarizer_provider
    chosen_supersession_model = supersession_model or fact_model or chosen_summarizer_model
    supersession_provider_key = chosen_supersession_provider
    if chosen_supersession_provider in providers:
        existing_model = providers[chosen_supersession_provider].get("model")
        if existing_model and existing_model != chosen_supersession_model:
            supersession_provider_key = f"{chosen_supersession_provider}-supersession"
            base = dict(providers[chosen_supersession_provider])
            base["model"] = chosen_supersession_model
            providers[supersession_provider_key] = base
    elif chosen_supersession_provider:
        _PROVIDER_TEMPLATES = {
            "openrouter": {"type": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
            "openai": {"type": "generic_openai", "base_url": "https://api.openai.com/v1", "api_key": openai_bearer_token or os.environ.get("OPENAI_API_KEY", "")},
            "ollama_native": {"type": "ollama_native", "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), "num_predict": 500, "force_json": True},
            "anthropic": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
        }
        tmpl = _PROVIDER_TEMPLATES.get(chosen_supersession_provider, {})
        if tmpl:
            providers[supersession_provider_key] = {**tmpl, "model": chosen_supersession_model}

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
            "provider": supersession_provider_key,
            "model": chosen_supersession_model,
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


def _messages_to_vc(q: MRCRQuestion) -> list[Message]:
    """Convert MRCR conversation messages into a flat Message list for VC ingestion.

    MRCR conversations are a single continuous dialogue (no sessions, no dates).
    """
    messages: list[Message] = []
    for m in q.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        messages.append(Message(role=role, content=content))
    return messages


def _dump_payload_log(
    *,
    question_id: str,
    cache_dir: Path,
    question_message: str,
    gold_answer: str,
    tags_matched: list[str],
    tokens_injected: int,
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
    """Dump the full HTTP request/response payloads for post-run analysis."""
    http_conversation = []
    for i in range(max(len(raw_requests), len(raw_responses))):
        if i < len(raw_requests):
            http_conversation.append({"step": i + 1, "direction": "REQUEST", "body": raw_requests[i]})
        if i < len(raw_responses):
            http_conversation.append({"step": i + 1, "direction": "RESPONSE", "body": raw_responses[i]})

    payload = {
        "_description": "Raw HTTP payloads for MRCR question.",
        "question_id": question_id,
        "question": question_message,
        "gold_answer": gold_answer,
        "hypothesis": hypothesis,
        "cached": cached,
        "summary": {
            "tags_matched": tags_matched,
            "tokens_injected": tokens_injected,
            "tool_calls_count": len(tool_calls),
            "continuation_count": continuation_count,
            "stop_reason": stop_reason,
            "tools_used": [tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool", "") if isinstance(tc, dict) else str(tc) for tc in tool_calls],
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = cache_dir / f"payload_log_{ts}.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("VC [%s]: payload log saved to %s", question_id, out_path)
    except Exception as e:
        logger.warning("VC [%s]: failed to save payload log: %s", question_id, e)


def _clear_compaction_state(cache_dir: str, question_id: str) -> None:
    """Clear compacted segments but keep TurnTagIndex (for recompact mode)."""
    import sqlite3
    db_path = Path(cache_dir) / "store.db"
    if not db_path.exists():
        logger.warning("No DB found at %s — nothing to clear", db_path)
        return
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM segment_tags")
    conn.execute("DELETE FROM segments")
    conn.execute("DELETE FROM tag_summaries")
    try:
        conn.execute("DELETE FROM segments_fts")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("DELETE FROM segments_fts_full")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("DELETE FROM chunk_embeddings")
    except sqlite3.OperationalError:
        pass
    conn.execute("UPDATE engine_state SET compacted_through = 0")
    conn.commit()
    conn.close()
    logger.info("VC [%s]: cleared compaction state, kept TurnTagIndex", question_id)


def run_vc_ingest_only(
    question: MRCRQuestion,
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
    curation_enabled: bool = False,
    curation_provider: str | None = None,
    curation_model: str | None = None,
    supersession: bool = False,
    supersession_provider: str | None = None,
    supersession_model: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Run only the ingest + compact phases (no reader query). Returns status dict."""
    q_cache_dir = _cache_dir_for(question.question_id, cache_dir)
    if fresh and q_cache_dir.exists():
        shutil.rmtree(q_cache_dir)
    q_cache_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = str(q_cache_dir)

    if recompact and not fresh:
        _clear_compaction_state(storage_dir, question.question_id)

    internal_openai_token = ""
    if tagger_provider == "openai" or (summarizer_provider or tagger_provider) == "openai":
        internal_openai_token = _resolve_openai_bearer_token()

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"mrcr-{question.question_id}",
        tagger_provider=tagger_provider,
        tagger_model=tagger_model,
        summarizer_provider=summarizer_provider,
        summarizer_model=summarizer_model,
        openai_bearer_token=internal_openai_token,
        tagger_mode=tagger_mode,
        fact_provider=fact_provider,
        fact_model=fact_model,
        curation_enabled=curation_enabled,
        curation_provider=curation_provider,
        curation_model=curation_model,
        supersession=supersession,
        supersession_provider=supersession_provider,
        supersession_model=supersession_model,
    )
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    messages = _messages_to_vc(question)

    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._compacted_through > 0

    if fully_cached:
        logger.info("VC [%s]: CACHE HIT — skipping ingest+compact", question.question_id)
        return {"turns_ingested": n_index_entries, "compaction_events": 0, "cached": True}

    turns_ingested = 0
    if n_index_entries == 0 or fresh:
        _ingest_start = time.time()

        def _progress(done: int, total: int, entry: object) -> None:
            if done % 20 == 0 or done == total:
                elapsed = time.time() - _ingest_start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info("  Ingest: %d/%d (%.1f turns/s)", done, total, rate)

        turns_ingested = engine.ingest_history(messages, progress_callback=_progress)
        logger.info("VC [%s]: ingested %d turns", question.question_id, turns_ingested)
    else:
        turns_ingested = n_index_entries

    compaction_events = 0
    while True:
        report = engine.compact_manual(messages)
        if report is None:
            break
        compaction_events += 1
        logger.info(
            "  Compaction #%d: %d segments, %d tokens freed",
            compaction_events, report.segments_compacted, report.tokens_freed,
        )

    engine._save_state(messages)

    # Supersession pass — deduplicate facts across sessions
    superseded_count = 0
    if supersession and engine._supersession_checker:
        t0 = time.time()
        all_facts = engine._store.query_facts(limit=9999)
        logger.info("VC [%s]: running supersession over %d facts...", question.question_id, len(all_facts))
        superseded_count = engine._supersession_checker.check_and_supersede(all_facts)
        logger.info("VC [%s]: supersession done — %d facts superseded in %.1fs",
                    question.question_id, superseded_count, time.time() - t0)

    return {
        "turns_ingested": turns_ingested,
        "compaction_events": compaction_events,
        "superseded_facts": superseded_count,
        "cached": False,
    }


def run_vc(
    question: MRCRQuestion,
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
    tagger_mode: str = "combined",
    fact_provider: str | None = None,
    fact_model: str | None = None,
    curation_enabled: bool = False,
    curation_provider: str | None = None,
    curation_model: str | None = None,
    supersession: bool = False,
    supersession_provider: str | None = None,
    supersession_model: str | None = None,
    cache_dir: Path | None = None,
    verbose_reasoning: bool = False,
) -> dict:
    """Run the VC pipeline for a single MRCR question.

    Steps: convert messages → ingest_history → compact_manual loop →
           on_message_inbound → query reader with tools

    Returns dict with: hypothesis, input_tokens, output_tokens, cost,
                       tags_matched, tokens_injected, compaction_events, timings.
    """
    # Resolve API key
    key_env = API_KEY_ENVS.get(reader_provider, "ANTHROPIC_API_KEY")
    if api_key:
        pass
    elif reader_provider in {"openai", "openai-codex", "openai-responses"}:
        api_key = _resolve_openai_bearer_token(reader_auth_mode)
    else:
        api_key = os.environ.get(key_env, "")

    chosen_summarizer_provider = summarizer_provider or tagger_provider
    needs_openai = tagger_provider == "openai" or chosen_summarizer_provider == "openai"
    internal_openai_token = ""
    if needs_openai:
        internal_openai_token = _resolve_openai_bearer_token(internal_openai_auth_mode)

    if not api_key:
        raise ValueError(f"No auth token found for reader provider {reader_provider}")

    timings: dict[str, float] = {}

    # 1. Build engine with stable cache dir
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
        session_id=f"mrcr-{question.question_id}",
        tagger_provider=tagger_provider,
        tagger_model=tagger_model,
        summarizer_provider=summarizer_provider,
        summarizer_model=summarizer_model,
        openai_bearer_token=internal_openai_token,
        tagger_mode=tagger_mode,
        fact_provider=fact_provider,
        fact_model=fact_model,
        curation_enabled=curation_enabled,
        curation_provider=curation_provider,
        curation_model=curation_model,
        supersession=supersession,
        supersession_provider=supersession_provider,
        supersession_model=supersession_model,
    )
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    # 2. Convert messages
    messages = _messages_to_vc(question)
    n_pairs = len(messages) // 2

    # 3. Check cache
    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._compacted_through > 0
    tags_only = not fully_cached and n_index_entries > 0

    if fully_cached:
        logger.info(
            "VC [%s]: CACHE HIT — %d turns indexed, compacted_through=%d",
            question.question_id, n_index_entries, engine._compacted_through,
        )
        compaction_events = -1
        timings["ingest_s"] = 0.0
        timings["compact_s"] = 0.0
        turns_ingested = n_index_entries
    elif tags_only:
        logger.info(
            "VC [%s]: RECOMPACT — %d turns indexed, re-running compaction",
            question.question_id, n_index_entries,
        )
        timings["ingest_s"] = 0.0
        turns_ingested = n_index_entries
    else:
        logger.info(
            "VC [%s]: %d messages (%d pairs), ~%d tokens",
            question.question_id, len(messages), n_pairs, question.tokens_est,
        )

        # Ingest
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
                "  Compaction #%d: %d segments, %d tokens freed, tags: %s",
                compaction_events, report.segments_compacted, report.tokens_freed, report.tags[:5],
            )
        timings["compact_s"] = round(time.time() - t0, 1)
        logger.info("VC [%s]: %d compaction events in %.1fs", question.question_id, compaction_events, timings["compact_s"])

        engine._save_state(messages)
        logger.info("VC [%s]: saved state to %s", question.question_id, storage_dir)

    # Supersession pass — deduplicate facts across sessions
    if supersession and engine._supersession_checker and not fully_cached:
        t0 = time.time()
        all_facts = engine._store.query_facts(limit=9999)
        logger.info("VC [%s]: running supersession over %d facts...", question.question_id, len(all_facts))
        superseded_count = engine._supersession_checker.check_and_supersede(all_facts)
        timings["supersession_s"] = round(time.time() - t0, 1)
        logger.info("VC [%s]: supersession done — %d facts superseded in %.1fs",
                    question.question_id, superseded_count, timings["supersession_s"])

    cached = fully_cached

    # Snapshot pre-reader telemetry
    pre_reader_telem = engine.get_telemetry().total()
    haiku_input = pre_reader_telem.input_tokens
    haiku_output = pre_reader_telem.output_tokens
    haiku_calls = pre_reader_telem.call_count

    # Persist/restore cost snapshot
    cost_snapshot_path = q_cache_dir / "cost_snapshot.json"
    pre_reader_provider = tagger_provider
    if haiku_input > 0 or haiku_output > 0:
        cost_snapshot_path.write_text(json.dumps({
            "haiku_input_tokens": haiku_input,
            "haiku_output_tokens": haiku_output,
            "haiku_calls": haiku_calls,
            "provider": pre_reader_provider,
        }))
    elif cost_snapshot_path.exists():
        snap = json.loads(cost_snapshot_path.read_text())
        haiku_input = snap.get("haiku_input_tokens", 0)
        haiku_output = snap.get("haiku_output_tokens", 0)
        haiku_calls = snap.get("haiku_calls", 0)
        pre_reader_provider = snap.get("provider", "anthropic")

    # 4. Retrieve context for the question
    t0 = time.time()
    assembled = engine.on_message_inbound(question.question_message, messages)
    timings["retrieve_s"] = round(time.time() - t0, 1)

    tags_matched = assembled.matched_tags
    prepend_text = assembled.prepend_text
    assembled_total_tokens = assembled.total_tokens

    prepend_tokens = len(prepend_text) // 4 if prepend_text else 0
    logger.info(
        "VC [%s]: assembled total=%d tokens, prepend=%d chars (~%dt), tags: %s",
        question.question_id, assembled_total_tokens, len(prepend_text), prepend_tokens, tags_matched[:10],
    )

    # Record pre-reader cost
    if haiku_input > 0 or haiku_output > 0:
        if pre_reader_provider == "anthropic":
            budget.record(
                label=f"vc_internal:{question.question_id}",
                model="claude-haiku-4-5-20251001",
                input_tokens=haiku_input,
                output_tokens=haiku_output,
            )

    # 5. Build reader prompt
    t0 = time.time()

    use_raw_history = engine._compacted_through == 0

    # MRCR question already contains the prepend instruction, but we reinforce it
    prepend_instruction = (
        f"IMPORTANT: You MUST prepend the following string exactly at the very "
        f"start of your response, before any other text: {question.random_string}"
    )

    if use_raw_history:
        # Pre-compaction: full raw history in system
        history_parts = []
        for m in question.messages:
            history_parts.append(f"{m['role'].capitalize()}: {m.get('content', '')}")
        system_prompt = (
            "I will give you a conversation history between you and a user. "
            "Please answer the question at the end based on the conversation.\n\n"
            f"{prepend_instruction}\n\n"
            "Conversation:\n\n" + "\n\n".join(history_parts)
        )
        user_prompt = question.question_message
    else:
        # Post-compaction: VC summaries in system
        context_hint = assembled.context_hint
        vc_summaries = "\n\n".join(assembled.tag_sections.values())
        system_parts = []
        if context_hint:
            system_parts.append(context_hint)
        system_parts.append(
            "I will give you a conversation history between you and a user. "
            "Please answer the question at the end based on the conversation.\n\n"
            f"{prepend_instruction}\n\n"
            "Conversation Summaries:\n\n"
            f"{vc_summaries}"
        )
        if assembled.facts_text:
            system_parts.append(assembled.facts_text)
        if assembled.conversation_history:
            conv_lines = [
                f"{msg.role.capitalize()}: {msg.content}"
                for msg in assembled.conversation_history
            ]
            system_parts.append("Recent Conversation:\n\n" + "\n\n".join(conv_lines))
        system_prompt = "\n\n".join(system_parts)
        user_prompt = question.question_message

    system_tokens_est = len(system_prompt) // 4
    user_tokens_est = len(user_prompt) // 4
    tokens_injected = system_tokens_est + user_tokens_est
    logger.info(
        "VC [%s]: reader prompt tokens est=%d (system=%d, user=%d), mode=%s",
        question.question_id, tokens_injected, system_tokens_est, user_tokens_est,
        "raw-history" if use_raw_history else "summary-history",
    )

    reader_api_url = API_URLS.get(reader_provider, "")
    require_tools = engine._compacted_through > 0

    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_prompt}],
        model=reader_model,
        system=system_prompt,
        max_tokens=8192,
        api_key=api_key,
        api_url=reader_api_url,
        temperature=0.0,
        force_tools=True,
        require_tools=require_tools,
        provider=reader_provider,
        extended_thinking=verbose_reasoning and reader_provider in ("anthropic", "openai-responses"),
    )

    timings["query_s"] = round(time.time() - t0, 1)

    # MRCR needs only the final round's text (not concatenated across tool rounds).
    # The tool loop accumulates text from all rounds, but intermediate rounds may
    # contain reasoning that would poison the random_string_to_prepend check.
    if loop_result.raw_responses:
        last_response = loop_result.raw_responses[-1]
        # Anthropic format
        content_blocks = last_response.get("content", [])
        final_texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        if not final_texts:
            # OpenAI/Gemini format
            choices = last_response.get("choices", [])
            if choices:
                final_texts = [choices[0].get("message", {}).get("content", "")]
            # Gemini format
            candidates = last_response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                final_texts = [p.get("text", "") for p in parts if "text" in p]
        hypothesis = "".join(final_texts)
    else:
        hypothesis = loop_result.text or ""

    # Strip reasoning preamble: if the model output reasoning before the
    # required prepend string, trim to the prepend string start.
    if question.random_string and question.random_string in hypothesis:
        idx = hypothesis.index(question.random_string)
        if idx > 0:
            hypothesis = hypothesis[idx:]
    reader_input = loop_result.input_tokens
    reader_output = loop_result.output_tokens
    tool_calls = loop_result.tool_calls or []

    entry = budget.record(
        label=f"vc_reader:{question.question_id}",
        model=reader_model,
        input_tokens=reader_input,
        output_tokens=reader_output,
    )

    total_cost = entry.cost_usd
    logger.info(
        "VC [%s]: reader %d in / %d out tokens, $%.4f, %.1fs, %d tool calls",
        question.question_id, reader_input, reader_output, total_cost, timings["query_s"], len(tool_calls),
    )

    # Dump payload log
    _dump_payload_log(
        question_id=question.question_id,
        cache_dir=q_cache_dir,
        question_message=question.question_message,
        gold_answer=question.answer,
        tags_matched=tags_matched,
        tokens_injected=tokens_injected,
        tool_calls=tool_calls,
        continuation_count=getattr(loop_result, "continuation_count", 0),
        stop_reason=getattr(loop_result, "stop_reason", ""),
        raw_requests=getattr(loop_result, "raw_requests", []),
        raw_responses=getattr(loop_result, "raw_responses", []),
        hypothesis=hypothesis,
        reader_input=reader_input,
        reader_output=reader_output,
        haiku_input=haiku_input,
        haiku_output=haiku_output,
        haiku_calls=haiku_calls,
        timings=timings,
        cached=cached,
    )

    return {
        "hypothesis": hypothesis,
        "input_tokens": reader_input,
        "output_tokens": reader_output,
        "cost": total_cost,
        "tags_matched": tags_matched,
        "tokens_injected": tokens_injected,
        "compaction_events": compaction_events,
        "assembled_total_tokens": assembled_total_tokens,
        "tool_calls": tool_calls,
        "timings": timings,
        "cached": cached,
        "haiku_input_tokens": haiku_input,
        "haiku_output_tokens": haiku_output,
        "haiku_calls": haiku_calls,
        "stop_reason": getattr(loop_result, "stop_reason", ""),
        "continuation_count": getattr(loop_result, "continuation_count", 0),
    }
