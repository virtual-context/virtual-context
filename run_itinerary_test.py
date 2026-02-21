#!/usr/bin/env python3
"""Run a custom question against a cached LongMemEval question's VC state.

Usage:
    # GPT-5.2 itinerary against cached Q12 (3fdac837)
    OPENAI_API_KEY=<key> ANTHROPIC_API_KEY=<key> \
        .venv/bin/python run_itinerary_test.py \
        --cache 3fdac837 \
        --question "Can you put together a complete itinerary for my Chicago trip?" \
        --provider openai --model gpt-5.2

    # Sonnet against cached Q12
    ANTHROPIC_API_KEY=<key> \
        .venv/bin/python run_itinerary_test.py \
        --cache 3fdac837 \
        --question "What hotel did we pick for Chicago?" \
        --provider anthropic --model claude-sonnet-4-5-20250929

    # Defaults: runs the Chicago itinerary question with GPT-5.2
    OPENAI_API_KEY=<key> ANTHROPIC_API_KEY=<key> \
        .venv/bin/python run_itinerary_test.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("custom_benchmark")

API_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
}
API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def build_engine_config(cache_dir: str, cache_id: str) -> dict:
    """Build VC config pointing at a cached LongMemEval question's store."""
    return {
        "version": "0.2",
        "session_id": f"bench-{cache_id}",
        "storage_root": cache_dir,
        "context_window": 65536,
        "token_counter": "estimate",
        "paging": {"enabled": True, "auto_promote": True, "auto_evict": True},
        "tag_generator": {
            "type": "llm",
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "max_tags": 10, "min_tags": 5,
            "prompt_mode": "detailed",
            "broad_heuristic_enabled": False,
            "temporal_heuristic_enabled": False,
            "context_bleed_threshold": 0,
        },
        "compaction": {
            "soft_threshold": 0.70, "hard_threshold": 0.85,
            "protected_recent_turns": 4, "overflow_buffer": 1.2,
            "summary_ratio": 0.15,
            "min_summary_tokens": 100, "max_summary_tokens": 800,
            "max_concurrent_summaries": 4,
        },
        "summarization": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500, "temperature": 0.3,
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
            "sqlite": {"path": f"{cache_dir}/store.db"},
            "filesystem": {"root": f"{cache_dir}/store"},
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
                    "min_overlap": 1, "max_results": 10,
                    "max_budget_fraction": 0.25, "include_related": True,
                },
            },
        },
        "cost_tracking": {
            "enabled": True,
            "pricing": {"anthropic": {"input_per_1k": 0.00025, "output_per_1k": 0.00125}},
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a custom question against a cached LongMemEval VC state.",
    )
    parser.add_argument(
        "--cache", default="3fdac837",
        help="Question ID / cache directory name (default: 3fdac837)",
    )
    parser.add_argument(
        "--question", "-q",
        default=(
            "Can you put together a complete itinerary for my Chicago trip based on "
            "everything we discussed \u2014 restaurants, hotel, transportation, and activities?"
        ),
        help="Custom question to ask",
    )
    parser.add_argument("--date", default="2024-11-17", help="Question date context")
    parser.add_argument("--provider", default="openai", choices=["anthropic", "openai", "gemini"])
    parser.add_argument("--model", default="gpt-5.2", help="Reader model ID")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-loops", type=int, default=5, help="Max tool loop continuations")
    parser.add_argument("--output", "-o", help="Output JSON path (default: cache_dir/<model>_result.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve API key
    key_env = API_KEY_ENVS.get(args.provider, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        print(f"ERROR: {key_env} not set")
        sys.exit(1)

    # Load engine from cache
    cache_dir = Path("benchmarks/longmemeval/cache") / args.cache
    if not cache_dir.exists():
        print(f"ERROR: cache not found: {cache_dir}")
        sys.exit(1)

    cfg_dict = build_engine_config(str(cache_dir), args.cache)
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    n_entries = len(engine._turn_tag_index.entries)
    logger.info("Engine: %d index entries, compacted_through=%d",
                n_entries, engine._compacted_through)

    if engine._compacted_through == 0:
        logger.warning("compacted_through=0 — cache may not have loaded. "
                       "Check session_id matches bench-%s", args.cache)

    # Retrieve context
    question_prompt = f"Current Date: {args.date}\nQuestion: {args.question}"
    t0 = time.time()
    assembled = engine.on_message_inbound(question_prompt, [])
    retrieve_s = time.time() - t0

    logger.info("Retrieved: %d tokens, %d tags in %.1fs: %s",
                assembled.total_tokens, len(assembled.matched_tags),
                retrieve_s, assembled.matched_tags)

    # Build prompts (same as benchmark harness)
    system_prompt = assembled.context_hint or ""
    vc_summaries = "\n\n".join(assembled.tag_sections.values())
    user_prompt = (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history.\n\n\n"
        "History Chats:\n\n"
        f"{vc_summaries}\n\n"
        f"Current Date: {args.date}\n"
        f"Question: {args.question}\n"
        f"Answer:"
    )

    if args.verbose:
        logger.info("System prompt (%d chars):\n%s", len(system_prompt), system_prompt[:500])
        logger.info("User prompt (%d chars):\n%s...", len(user_prompt), user_prompt[:500])

    # Query with tools
    api_url = API_URLS.get(args.provider, "")
    logger.info("Querying %s / %s...", args.provider, args.model)
    t0 = time.time()
    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_prompt}],
        model=args.model,
        system=system_prompt,
        max_tokens=args.max_tokens,
        api_key=api_key,
        api_url=api_url,
        temperature=0.0,
        force_tools=True,
        provider=args.provider,
        max_loops=args.max_loops,
    )
    query_s = time.time() - t0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Cache:         {args.cache}")
    print(f"Provider:      {args.provider} / {args.model}")
    print(f"Input tokens:  {loop_result.input_tokens}")
    print(f"Output tokens: {loop_result.output_tokens}")
    print(f"Continuations: {loop_result.continuation_count}")
    print(f"Stop reason:   {loop_result.stop_reason}")
    print(f"Query time:    {query_s:.1f}s")
    print(f"Tool calls:    {len(loop_result.tool_calls)}")

    for tc in loop_result.tool_calls:
        preview = tc.result_json[:100] if tc.result_json else ""
        print(f"  {tc.tool_name}({tc.tool_input}) -> {preview}... ({tc.duration_ms}ms)")

    print(f"\n--- Hypothesis ---\n{loop_result.text}")

    # Verify inject_context fix: check system prompts across requests
    print("\n--- System Prompt Verification ---")
    for i, req in enumerate(loop_result.raw_requests):
        msgs = req.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            sys_content = msgs[0].get("content", "")
            has_vc = "<virtual-context>" in sys_content
            vc_size = len(sys_content.split("<virtual-context>")[1].split("</virtual-context>")[0]) if has_vc else 0
            print(f"  Request {i}: VC block={has_vc}, VC size={vc_size} chars, total system={len(sys_content)} chars")
        elif "system" in req:
            sys_val = req["system"]
            if isinstance(sys_val, str):
                has_vc = "<virtual-context>" in sys_val
                vc_size = len(sys_val.split("<virtual-context>")[1].split("</virtual-context>")[0]) if has_vc else 0
                print(f"  Request {i}: system key, VC block={has_vc}, VC size={vc_size} chars")
            else:
                print(f"  Request {i}: system key (list format), {len(sys_val)} entries")
        else:
            print(f"  Request {i}: no system prompt")

    # Save results
    out_path = args.output or str(cache_dir / f"{args.model.replace('.', '_')}_result.json")
    payload = {
        "question": args.question,
        "provider": args.provider,
        "model": args.model,
        "cache": args.cache,
        "hypothesis": loop_result.text,
        "input_tokens": loop_result.input_tokens,
        "output_tokens": loop_result.output_tokens,
        "continuation_count": loop_result.continuation_count,
        "stop_reason": loop_result.stop_reason,
        "query_s": round(query_s, 1),
        "tool_calls": [
            {"tool": tc.tool_name, "input": tc.tool_input,
             "result": tc.result_json, "duration_ms": tc.duration_ms}
            for tc in loop_result.tool_calls
        ],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
