"""CLI orchestrator for BEAM benchmark evaluation with VC.

API keys live in benchmarks/beam/.env (gitignored).
Load with: source benchmarks/beam/.env or export vars before running.

Usage:
    python -m benchmarks.beam [flags]

Flags:
  DATA
    --beam-root PATH          Path to cloned BEAM repo (default: ~/projects/BEAM)
    --chat-size SIZE          100K | 500K | 1M | 10M (default: 100K)
    --start N                 First conversation index (default: 0)
    --end N                   Last conversation index, exclusive (default: all)
    --conversations ID [ID …] Run only these conversation IDs
    --categories CAT [CAT …]  Filter probing-question categories
    --question-id ID [ID …]   Run only these question IDs (e.g. 100K_1_abstention_0).
                              Can be combined with --categories.

  VC ENGINE
    --context-window N        Context window in tokens (default: 65536)
    --reader-model MODEL      Reader model for question answering (default: claude-sonnet-4-20250514)
    --reader-provider NAME    anthropic | openai | openrouter (default: anthropic)
    --tagger-model MODEL      Tagger/summarizer model for ingestion (REQUIRED, no default)
    --tagger-provider NAME    Provider for tagger model (default: openrouter)
    --summarizer-provider NAME  Override tagger provider for summarizer only
    --summarizer-model MODEL    Override tagger model for summarizer only
    --tagger-mode MODE        split | unified (default: split)
    --fact-provider NAME      Override provider for fact extraction
    --fact-model MODEL        Override model for fact extraction

  CACHE CONTROL
    --clear-cache             ** DESTRUCTIVE ** Delete the entire cache directory for each
                              conversation and re-ingest from scratch. This removes all
                              stored segments, tags, facts, compaction state, and payload
                              logs. You will pay full ingestion cost again.
    --recompact               Keep cached tags and facts, re-run compaction only. Cheaper
                              than --clear-cache but still re-processes segments.
    --cache-dir PATH          Override default cache location (default: benchmarks/beam/cache/)
    --ingest-only             Run ingestion and compaction only — skip question answering.

  JUDGING
    --judge                   Enable LLM-as-judge scoring after each question
    --judge-model MODEL       Model for judging (default: claude-sonnet-4-20250514)
    --judge-provider NAME     Provider for judge model (default: anthropic)
    --event-ordering-judge-mode MODE
                              full | fast (default: full). `fast` uses a local
                              lexical alignment instead of pairwise LLM judging
                              for event_ordering questions.

  AUTOPSY
    --no-autopsy-report       Skip autopsy report generation after the run
    --autopsy-output-prefix P Output prefix for autopsy files (without extension)

  OUTPUT
    --budget N                Max spend in USD (default: 50.0)
    -o, --output PATH         Write results JSON to this path
    -v, --verbose             Enable DEBUG-level logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Auto-load API keys from benchmarks/beam/.env (gitignored)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip()

from benchmarks.longmemeval.cost import BudgetTracker
from .dataset import load_conversations, BEAMConversation
from .vc_runner import ingest_conversation, query_question

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BEAM benchmark evaluation with Virtual Context.",
    )
    # Data
    p.add_argument("--beam-root", type=str, default=os.path.expanduser("~/projects/BEAM"),
                    help="Path to BEAM repo root (default: ~/projects/BEAM)")
    p.add_argument("--chat-size", type=str, default="100K",
                    choices=["100K", "500K", "1M", "10M"],
                    help="Chat size to evaluate (default: 100K)")
    p.add_argument("--start", type=int, default=0, help="Start conversation index")
    p.add_argument("--end", type=int, default=None, help="End conversation index (exclusive)")
    p.add_argument("--conversations", nargs="+", default=None,
                    help="Specific conversation IDs to run")
    p.add_argument("--categories", nargs="+", default=None,
                    help="Filter question categories (e.g. abstention information_extraction)")
    p.add_argument("--question-id", nargs="+", default=None,
                    help="Run only these question IDs (e.g. 100K_1_abstention_0)")

    # VC config
    p.add_argument("--context-window", type=int, default=65536)
    p.add_argument("--reader-model", type=str, default="claude-sonnet-4-20250514")
    p.add_argument("--reader-provider", type=str, default="anthropic")
    p.add_argument("--tagger-model", type=str, required=True,
                     help="Model for tagging/summarization (e.g. claude-haiku-4-5-20251001)")
    p.add_argument("--tagger-provider", type=str, default="openrouter")
    p.add_argument("--summarizer-provider", type=str, default=None)
    p.add_argument("--summarizer-model", type=str, default=None)
    p.add_argument("--tagger-mode", type=str, default="split")
    p.add_argument("--fact-provider", type=str, default=None)
    p.add_argument("--fact-model", type=str, default=None)

    # Cache control
    p.add_argument("--clear-cache", action="store_true", dest="clear_cache",
                    help="DESTRUCTIVE: Delete the entire cache directory for each conversation "
                         "and re-ingest from scratch. Removes all stored segments, tags, facts, "
                         "compaction state, and payload logs.")
    p.add_argument("--recompact", action="store_true", help="Keep tags, re-run compaction")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--ingest-only", action="store_true", help="Ingest + compact only, no queries")
    p.add_argument(
        "--require-fully-cached",
        action="store_true",
        help="Fail instead of ingesting or compacting when the conversation cache is not already fully compacted.",
    )

    # Judging
    p.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring")
    p.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514")
    p.add_argument("--judge-provider", type=str, default="anthropic")
    p.add_argument("--event-ordering-judge-mode", type=str, default="full",
                   choices=["full", "fast"])

    # Autopsy report
    p.add_argument("--no-autopsy-report", action="store_true",
                    help="Disable Autopsy report sidecar generation.")
    p.add_argument("--autopsy-output-prefix", type=str, default=None,
                    help="Optional output prefix for Autopsy report files (without extension).")

    # Budget and output
    p.add_argument("--budget", type=float, default=50.0, help="Max spend in USD")
    p.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load conversations
    logger.info("Loading BEAM %s conversations from %s", args.chat_size, args.beam_root)
    conversations = load_conversations(args.beam_root, args.chat_size)
    logger.info("Loaded %d conversations", len(conversations))

    # Filter by index range or specific IDs
    if args.conversations:
        conversations = [c for c in conversations if c.conv_id in args.conversations]
    else:
        end = args.end if args.end is not None else len(conversations)
        conversations = conversations[args.start:end]

    if not conversations:
        logger.error("No conversations to process")
        sys.exit(1)

    logger.info("Processing %d conversations", len(conversations))

    budget = BudgetTracker(budget=args.budget)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    results: list[dict] = []
    total_questions = 0
    total_judged = 0
    start_time = time.time()

    for conv_idx, conv in enumerate(conversations):
        logger.info(
            "=== Conversation %d/%d: %s (%d questions, ~%dK tokens) ===",
            conv_idx + 1, len(conversations), conv.conv_id,
            len(conv.questions), conv.est_tokens // 1000,
        )

        # Ingest
        try:
            engine, messages, ingest_stats = ingest_conversation(
                conv,
                context_window=args.context_window,
                clear_cache=args.clear_cache,
                recompact=args.recompact,
                require_fully_cached=args.require_fully_cached,
                tagger_provider=args.tagger_provider,
                tagger_model=args.tagger_model,
                summarizer_provider=args.summarizer_provider,
                summarizer_model=args.summarizer_model,
                tagger_mode=args.tagger_mode,
                fact_provider=args.fact_provider,
                fact_model=args.fact_model,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error("Failed to ingest %s: %s", conv.conv_id, e, exc_info=True)
            continue

        if args.ingest_only:
            logger.info("Ingest-only mode -- skipping queries for %s", conv.conv_id)
            results.append({
                "conv_id": conv.conv_id,
                "ingest_stats": ingest_stats,
                "questions": [],
            })
            continue

        # Filter questions by category and/or ID
        questions = conv.questions
        if args.categories:
            questions = [q for q in questions if q.category in args.categories]
        if args.question_id:
            questions = [q for q in questions if q.question_id in args.question_id]

        # Query each question
        conv_results: list[dict] = []
        for q_idx, question in enumerate(questions):
            logger.info(
                "  Q %d/%d [%s]: %s",
                q_idx + 1, len(questions), question.category,
                question.question[:80] + ("..." if len(question.question) > 80 else ""),
            )

            try:
                qr = query_question(
                    engine, messages, question, conv, budget,
                    reader_model=args.reader_model,
                    reader_provider=args.reader_provider,
                    cache_dir=cache_dir,
                )
            except Exception as e:
                logger.error("  Failed: %s", e, exc_info=args.verbose)
                qr = {"hypothesis": f"ERROR: {e}", "error": True}

            q_result = {
                "question_id": question.question_id,
                "category": question.category,
                "question": question.question,
                "hypothesis": qr.get("hypothesis", ""),
                "ideal_response": question.ideal_response,
                "rubric": question.rubric,
                "difficulty": question.difficulty,
                "timings": qr.get("timings", {}),
                "tags_matched": qr.get("tags_matched", []),
                "tokens_injected": qr.get("tokens_injected", 0),
                "tool_calls": qr.get("tool_calls", []),
            }

            # Judge if requested
            if args.judge and not qr.get("error"):
                try:
                    from .judge import judge_answer
                    judge_api_key = os.environ.get(
                        "OPENAI_API_KEY" if args.judge_provider == "openai" else "ANTHROPIC_API_KEY",
                        "",
                    )
                    judgment = judge_answer(
                        question, qr["hypothesis"],
                        model=args.judge_model,
                        api_key=judge_api_key,
                        provider=args.judge_provider,
                        event_ordering_mode=args.event_ordering_judge_mode,
                    )
                    q_result["judgment"] = judgment
                    total_judged += 1
                    logger.info("  Score: %.2f (%d/%d criteria met)",
                                judgment["score"],
                                len(judgment["criteria_met"]),
                                len(question.rubric))
                except Exception as e:
                    logger.error("  Judge failed: %s", e)
                    q_result["judgment"] = {"score": None, "error": str(e)}

            conv_results.append(q_result)
            total_questions += 1

            logger.info("  -> %s", qr.get("hypothesis", "")[:100])

        results.append({
            "conv_id": conv.conv_id,
            "ingest_stats": ingest_stats,
            "questions": conv_results,
        })

        # Save incrementally
        if args.output:
            _save_results(args, results, budget, start_time)

    # Final save
    if args.output:
        _save_results(args, results, budget, start_time)

    # Autopsy report
    if args.output and not args.no_autopsy_report:
        from .autopsy_report import write_autopsy_reports
        out_path = Path(args.output)
        output_data = json.loads(out_path.read_text())
        autopsy_json, autopsy_md = write_autopsy_reports(
            results_data=output_data,
            results_output_path=out_path,
            autopsy_output_prefix=args.autopsy_output_prefix,
            cache_dir=cache_dir,
        )
        logger.info("Autopsy report (JSON): %s", autopsy_json)
        logger.info("Autopsy report (Markdown): %s", autopsy_md)

    # Print summary
    _print_summary(results, budget, start_time)


def _save_results(args: argparse.Namespace, results: list[dict],
                  budget: BudgetTracker, start_time: float) -> None:
    """Save results JSON to disk."""
    output = {
        "benchmark": "beam",
        "chat_size": args.chat_size,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "tagger_model": args.tagger_model,
            "tagger_provider": args.tagger_provider,
            "context_window": args.context_window,
            "judge": args.judge,
            "event_ordering_judge_mode": args.event_ordering_judge_mode,
        },
        "conversations": results,
        "summary": _compute_summary(results),
        "elapsed_s": round(time.time() - start_time, 1),
        "budget_spent": budget.spent,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results saved to %s", args.output)


def _compute_summary(results: list[dict]) -> dict:
    """Compute aggregate scores from results."""
    by_category: dict[str, list[float]] = {}
    all_scores: list[float] = []

    for conv in results:
        for q in conv.get("questions", []):
            j = q.get("judgment")
            if j and j.get("score") is not None:
                score = j["score"]
                all_scores.append(score)
                cat = q["category"]
                by_category.setdefault(cat, []).append(score)

    per_category = {}
    for cat, scores in sorted(by_category.items()):
        per_category[cat] = {
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "count": len(scores),
        }

    return {
        "overall_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else None,
        "total_questions": sum(len(c.get("questions", [])) for c in results),
        "total_judged": len(all_scores),
        "per_category": per_category,
    }


def _print_summary(results: list[dict], budget: BudgetTracker, start_time: float) -> None:
    """Print human-readable summary to stdout."""
    summary = _compute_summary(results)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("BEAM Benchmark Results")
    print("=" * 60)
    print(f"Conversations: {len(results)}")
    print(f"Questions answered: {summary['total_questions']}")
    print(f"Questions judged: {summary['total_judged']}")

    if summary.get("overall_score") is not None:
        print(f"Overall score: {summary['overall_score']:.3f}")
        print("\nPer category:")
        for cat, info in summary["per_category"].items():
            print(f"  {cat:30s} {info['avg_score']:.3f}  (n={info['count']})")

    print(f"\nElapsed: {elapsed:.0f}s")
    print(f"Cost: ${budget.spent:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
