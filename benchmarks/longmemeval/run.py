"""CLI entry point for running the LongMemEval benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .baseline import run_baseline
from .cost import BudgetTracker
from .dataset import download_dataset, load_dataset, select_questions
from .judge import judge_answer
from .vc_runner import clear_cache, run_vc

logger = logging.getLogger(__name__)


def _incremental_save(results: list[dict], args, budget) -> None:
    """Write results after each question so progress survives process kills."""
    if args.output is None:
        args.output = f"longmemeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = Path(args.output)
    data = {
        "benchmark": "longmemeval",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "config": {
            "context_window": args.context_window,
            "baseline_model": args.baseline_model,
            "reader_model": args.reader_model,
            "judge_model": args.judge_model,
        },
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 4),
        "questions": results,
        "partial": True,
    }
    out_path.write_text(json.dumps(data, indent=2, default=str))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # Force stderr for all logging, with explicit flush
    handler = logging.StreamHandler(stream=__import__('sys').stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # Quiet down noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark: compare VC vs full-context baseline",
        prog="python -m benchmarks.longmemeval.run",
    )
    parser.add_argument("--budget", type=float, default=5.0, help="Max spend in USD (default: $5)")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to run (default: 5)")
    parser.add_argument("--questions", nargs="+", help="Specific question IDs to run")
    parser.add_argument("--categories", nargs="+", help="Question categories to include")
    parser.add_argument("--context-window", type=int, default=65536, help="VC context window (default: 65536)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto-generated)")
    parser.add_argument("--baseline-model", type=str, default="claude-sonnet-4-5-20250929", help="Baseline reader model")
    parser.add_argument("--reader-model", type=str, default="claude-sonnet-4-5-20250929", help="VC reader model")
    parser.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001", help="Judge model")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline, run VC only")
    parser.add_argument("--skip-vc", action="store_true", help="Skip VC, run baseline only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--fresh", action="store_true", help="Clear cached ingestion+compaction (re-ingest from scratch)")
    parser.add_argument("--recompact", action="store_true", help="Keep cached tags, re-run compaction only (faster than --fresh)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all VC caches and exit")
    parser.add_argument("--summarizer-model", type=str, default=None, help="Override summarization model (e.g. gpt-4o-mini)")
    parser.add_argument("--reader-provider", type=str, default="anthropic",
                        choices=["anthropic", "openai", "gemini"],
                        help="LLM provider for reader model (default: anthropic)")
    parser.add_argument("--download-only", action="store_true", help="Just download the dataset and exit")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # Cache management
    if args.clear_cache:
        cleared = clear_cache()
        print(f"Cleared {cleared} cached question(s).")
        return

    # Download dataset
    download_dataset()
    if args.download_only:
        print("Dataset downloaded. Exiting.")
        return

    # Load and select questions
    dataset = load_dataset()
    questions = select_questions(
        dataset,
        count=args.count,
        categories=args.categories,
        question_ids=args.questions,
    )

    if not questions:
        print("No questions selected. Check your filters.")
        return

    budget = BudgetTracker(budget=args.budget)
    results: list[dict] = []

    print(f"\n{'='*70}")
    print(f"LongMemEval Smoke Test — {len(questions)} questions, ${args.budget:.2f} budget")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        print(f"\n--- Question {i}/{len(questions)}: {q.question_id} ({q.question_type}) ---")
        print(f"  Q: {q.question[:120]}...")
        print(f"  A: {q.answer[:120]}...")
        print(f"  Haystack: ~{q.haystack_tokens_est:,} tokens, {len(q.haystack_sessions)} sessions")

        # Budget check
        est_cost = budget.estimate_question_cost(q.haystack_tokens_est)
        if not budget.can_afford(est_cost):
            print(f"  SKIP: estimated cost ${est_cost:.2f} exceeds remaining budget ${budget.remaining:.2f}")
            break

        entry: dict = {
            "question_id": q.question_id,
            "question_type": q.question_type,
            "question": q.question,
            "answer": q.answer,
            "haystack_tokens_est": q.haystack_tokens_est,
            "n_sessions": len(q.haystack_sessions),
        }

        # Run baseline
        if not args.skip_baseline:
            try:
                print(f"  Running baseline ({args.baseline_model})...")
                t0 = time.time()
                baseline_result = run_baseline(q, budget, model=args.baseline_model)
                baseline_result["elapsed_s"] = round(time.time() - t0, 1)
                print(f"  Baseline: \"{baseline_result['hypothesis'][:100]}...\"  (${baseline_result['cost']:.4f})")

                # Judge baseline
                baseline_judge = judge_answer(
                    question=q.question,
                    answer=q.answer,
                    hypothesis=baseline_result["hypothesis"],
                    question_type=q.question_type,
                    budget=budget,
                    label=f"baseline:{q.question_id}",
                    model=args.judge_model,
                )
                baseline_result["correct"] = baseline_judge["correct"]
                baseline_result["judge_explanation"] = baseline_judge["explanation"]
                print(f"  Baseline judge: {'CORRECT' if baseline_judge['correct'] else 'WRONG'}")

                entry["baseline"] = baseline_result
            except Exception as e:
                logger.error("Baseline failed for %s: %s", q.question_id, e)
                entry["baseline"] = {"error": str(e)}

        # Run VC
        if not args.skip_vc:
            try:
                print(f"  Running VC (window={args.context_window})...")
                t0 = time.time()
                vc_result = run_vc(
                    q, budget,
                    context_window=args.context_window,
                    reader_model=args.reader_model,
                    verbose=args.verbose,
                    fresh=args.fresh,
                    recompact=args.recompact,
                    summarizer_model=args.summarizer_model,
                    reader_provider=args.reader_provider,
                )
                vc_result["elapsed_s"] = round(time.time() - t0, 1)
                print(f"  VC: \"{vc_result['hypothesis'][:100]}...\"  (${vc_result['cost']:.4f})")
                print(f"      tags: {vc_result['tags_matched'][:5]}, {vc_result['tokens_injected']}t injected, {vc_result['compaction_events']} compactions")

                # Judge VC
                vc_judge = judge_answer(
                    question=q.question,
                    answer=q.answer,
                    hypothesis=vc_result["hypothesis"],
                    question_type=q.question_type,
                    budget=budget,
                    label=f"vc:{q.question_id}",
                    model=args.judge_model,
                )
                vc_result["correct"] = vc_judge["correct"]
                vc_result["judge_explanation"] = vc_judge["explanation"]
                print(f"  VC judge: {'CORRECT' if vc_judge['correct'] else 'WRONG'}")

                entry["vc"] = vc_result
            except Exception as e:
                logger.error("VC failed for %s: %s", q.question_id, e, exc_info=args.verbose)
                entry["vc"] = {"error": str(e)}

        results.append(entry)

        # Incremental save — don't lose judged results if process is killed
        _incremental_save(results, args, budget)

        print(f"  Budget: ${budget.spent:.4f} / ${budget.budget:.2f} spent")

    # Compute summary
    summary = _compute_summary(results, args)

    output_data = {
        "benchmark": "longmemeval",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "config": {
            "context_window": args.context_window,
            "baseline_model": args.baseline_model,
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "judge_model": args.judge_model,
        },
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 4),
        "questions": results,
        "summary": summary,
        "cost_breakdown": budget.summary(),
    }

    # Write output
    if args.output is None:
        args.output = f"longmemeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\n{'='*70}")
    print(f"Results saved to: {out_path}")
    _print_summary(summary, budget)
    print(f"{'='*70}\n")


def _compute_summary(results: list[dict], args: argparse.Namespace) -> dict:
    """Compute accuracy summary from results."""
    summary: dict = {}

    for method in ["baseline", "vc"]:
        if (method == "baseline" and args.skip_baseline) or (method == "vc" and args.skip_vc):
            continue

        correct = 0
        total = 0
        per_cat: dict[str, dict] = {}

        for r in results:
            if method not in r or "error" in r[method]:
                continue
            total += 1
            is_correct = r[method].get("correct", False)
            if is_correct:
                correct += 1

            cat = r["question_type"]
            if cat not in per_cat:
                per_cat[cat] = {"correct": 0, "total": 0}
            per_cat[cat]["total"] += 1
            if is_correct:
                per_cat[cat]["correct"] += 1

        summary[f"{method}_accuracy"] = correct / total if total > 0 else 0.0
        summary[f"{method}_correct"] = correct
        summary[f"{method}_total"] = total
        summary[f"{method}_per_category"] = {
            cat: {"accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0, **d}
            for cat, d in per_cat.items()
        }

    return summary


def _print_summary(summary: dict, budget: BudgetTracker) -> None:
    """Print a human-readable summary."""
    print(f"\n  Total cost: ${budget.spent:.4f} / ${budget.budget:.2f}")

    for method in ["baseline", "vc"]:
        key = f"{method}_accuracy"
        if key in summary:
            correct = summary[f"{method}_correct"]
            total = summary[f"{method}_total"]
            acc = summary[key]
            print(f"  {method.upper():>10}: {correct}/{total} correct ({acc:.0%})")

            per_cat = summary.get(f"{method}_per_category", {})
            for cat, d in per_cat.items():
                mark = "+" if d["correct"] > 0 else "-"
                print(f"    {mark} {cat}: {d['correct']}/{d['total']}")


if __name__ == "__main__":
    main()
