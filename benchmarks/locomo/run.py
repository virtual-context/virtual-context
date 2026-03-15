"""LocOMo benchmark CLI: compare VC vs full-context baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean

from benchmarks.longmemeval.cost import BudgetTracker
from .dataset import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    LoCoMoQuestion,
    load_dataset,
    select_conversations,
    select_questions,
)
from .scoring import judge_question, score_question
from .vc_runner import ingest_conversation, query_question
from .baseline import run_baseline

logger = logging.getLogger(__name__)


def _incremental_save(results: list[dict], output_path: Path, args, budget: BudgetTracker) -> None:
    """Save results after each question for crash recovery."""
    data = {
        "benchmark": "locomo",
        "config": {
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "baseline_model": args.baseline_model,
            "baseline_provider": args.baseline_provider,
            "tagger_model": args.tagger_model,
            "tagger_provider": args.tagger_provider,
        },
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 6),
        "questions": results,
    }
    try:
        output_path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.warning("Failed to save results: %s", e)


def _compute_summary(results: list[dict]) -> dict:
    """Compute per-category F1 + judge accuracy summary."""
    summary: dict = {}
    for method in ("baseline", "vc"):
        per_cat_f1: dict[int, list[float]] = {}
        per_cat_judge: dict[int, list[bool]] = {}
        all_f1: list[float] = []
        all_judge: list[bool] = []
        for r in results:
            if method not in r or "error" in r.get(method, {}):
                continue
            f1_val = r[method].get("f1", 0.0)
            all_f1.append(f1_val)
            cat = r["category"]
            per_cat_f1.setdefault(cat, []).append(f1_val)

            judge_val = r[method].get("judge")
            if judge_val is not None:
                all_judge.append(judge_val)
                per_cat_judge.setdefault(cat, []).append(judge_val)

        if all_f1:
            summary[f"{method}_f1_mean"] = round(mean(all_f1), 4)
            summary[f"{method}_total"] = len(all_f1)
            cat_data: dict = {}
            for cat, scores in sorted(per_cat_f1.items()):
                entry = {
                    "name": CATEGORY_NAMES.get(cat, "?"),
                    "f1_mean": round(mean(scores), 4),
                    "count": len(scores),
                }
                if cat in per_cat_judge:
                    j = per_cat_judge[cat]
                    entry["judge_accuracy"] = round(sum(j) / len(j), 4)
                    entry["judge_count"] = len(j)
                cat_data[cat] = entry
            summary[f"{method}_per_category"] = cat_data

        if all_judge:
            summary[f"{method}_judge_accuracy"] = round(sum(all_judge) / len(all_judge), 4)
            summary[f"{method}_judge_total"] = len(all_judge)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LocOMo benchmark: VC vs full-context baseline",
        prog="python -m benchmarks.locomo",
    )

    # Budget & output
    parser.add_argument("--budget", type=float, default=10.0, help="Max spend in USD")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Dataset selection
    parser.add_argument("--conversations", nargs="+", help="Specific conv IDs (e.g. conv-26)")
    parser.add_argument("--categories", nargs="+", type=int, help="QA categories (1-5)")
    parser.add_argument("--questions", nargs="+", help="Specific question IDs")
    parser.add_argument("--count", type=int, default=None, help="Max questions per conversation")
    parser.add_argument("--data-file", type=str, default=None, help="Path to locomo10.json")

    # VC config
    parser.add_argument("--context-window", type=int, default=65536)
    parser.add_argument("--fresh", action="store_true", help="Clear cache and re-ingest")
    parser.add_argument("--recompact", action="store_true", help="Re-run compaction only")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--tagger-provider", type=str, default="openrouter",
                        choices=["anthropic", "openai", "ollama_native", "openrouter"])
    parser.add_argument("--tagger-model", type=str, default="xiaomi/mimo-v2-flash")
    parser.add_argument("--tagger-mode", type=str, default="split", choices=["combined", "split"])
    parser.add_argument("--fact-provider", type=str, default="openrouter")
    parser.add_argument("--fact-model", type=str, default="xiaomi/mimo-v2-flash")
    parser.add_argument("--summarizer-provider", type=str, default="openrouter")
    parser.add_argument("--summarizer-model", type=str, default="xiaomi/mimo-v2-flash")
    parser.add_argument("--supersession", action="store_true")
    parser.add_argument("--curation", action="store_true")
    parser.add_argument("--curation-provider", type=str, default=None)
    parser.add_argument("--curation-model", type=str, default=None)

    # Reader / baseline / judge models
    parser.add_argument("--reader-model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--reader-provider", type=str, default="gemini",
                        choices=["anthropic", "openai", "gemini", "openrouter"])
    parser.add_argument("--baseline-model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--baseline-provider", type=str, default="gemini",
                        choices=["anthropic", "openai", "gemini", "openrouter"])

    # Judge
    parser.add_argument("--judge", action="store_true", help="Run LLM judge alongside F1")
    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--judge-provider", type=str, default="gemini",
                        choices=["gemini", "anthropic", "openrouter"])

    # Run modes
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-vc", action="store_true")
    parser.add_argument("--ingest-only", action="store_true", help="Ingest+compact only, no reader")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed questions")
    parser.add_argument("--explain", action="store_true", help="Ask reader to explain its reasoning")

    args = parser.parse_args(argv)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load dataset
    data_path = Path(args.data_file) if args.data_file else None
    dataset = load_dataset(data_path)
    conversations = select_conversations(dataset, conv_ids=args.conversations)

    # Budget
    budget = BudgetTracker(budget=args.budget)

    # Output path
    output_path = Path(args.output) if args.output else Path(f"locomo_results.json")

    # Resume: load existing results
    completed_qids: set[str] = set()
    results: list[dict] = []
    if args.resume and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            results = existing.get("questions", [])
            completed_qids = {r["question_id"] for r in results if "error" not in r.get("vc", r.get("baseline", {}))}
            logger.info("Resumed: %d questions already completed", len(completed_qids))
        except Exception:
            pass

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Count total questions
    total_qs = 0
    for conv in conversations:
        qs = select_questions(conv, categories=args.categories, count=args.count, question_ids=args.questions)
        total_qs += len(qs)

    print(f"\n{'=' * 70}")
    print(f"LocOMo Benchmark — {len(conversations)} conversations, {total_qs} questions, ${args.budget:.2f} budget")
    print(f"{'=' * 70}\n")

    if args.ingest_only:
        # Ingest-only mode
        for conv in conversations:
            print(f"\n--- Ingesting {conv.conv_id}: {conv.total_turns} turns, ~{conv.est_tokens // 1000}K tokens ---")
            engine, messages, stats = ingest_conversation(
                conv,
                context_window=args.context_window,
                fresh=args.fresh,
                recompact=args.recompact,
                tagger_provider=args.tagger_provider,
                tagger_model=args.tagger_model,
                summarizer_provider=args.summarizer_provider,
                summarizer_model=args.summarizer_model,
                tagger_mode=args.tagger_mode,
                fact_provider=args.fact_provider,
                fact_model=args.fact_model,
                cache_dir=cache_dir,
                supersession=args.supersession,
                curation_enabled=args.curation,
                curation_provider=args.curation_provider,
                curation_model=args.curation_model,
            )
            engine.close()
            print(f"  {conv.conv_id}: {stats['turns_ingested']} turns, {stats['compaction_events']} compactions")
        print(f"\n{'=' * 70}")
        print(f"Ingest complete: {len(conversations)} conversations")
        print(f"{'=' * 70}")
        return

    # Main loop: conversation-outer, question-inner
    q_num = 0
    for conv in conversations:
        qs = select_questions(conv, categories=args.categories, count=args.count, question_ids=args.questions)
        if not qs:
            continue

        print(f"\n--- {conv.conv_id}: {conv.speaker_a} & {conv.speaker_b}, "
              f"{len(conv.sessions)} sessions, {len(qs)} questions ---")

        # Ingest conversation once for VC
        engine = None
        messages = None
        if not args.skip_vc:
            engine, messages, stats = ingest_conversation(
                conv,
                context_window=args.context_window,
                fresh=args.fresh,
                recompact=args.recompact,
                tagger_provider=args.tagger_provider,
                tagger_model=args.tagger_model,
                summarizer_provider=args.summarizer_provider,
                summarizer_model=args.summarizer_model,
                tagger_mode=args.tagger_mode,
                fact_provider=args.fact_provider,
                fact_model=args.fact_model,
                cache_dir=cache_dir,
                supersession=args.supersession,
                curation_enabled=args.curation,
                curation_provider=args.curation_provider,
                curation_model=args.curation_model,
            )

        for question in qs:
            q_num += 1

            if question.question_id in completed_qids:
                print(f"  [{q_num}/{total_qs}] {question.question_id} — SKIP (resumed)")
                continue

            if budget.spent >= args.budget:
                print(f"  Budget exhausted (${budget.spent:.4f} / ${args.budget:.2f})")
                break

            cat_name = CATEGORY_NAMES.get(question.category, "?")
            print(f"  [{q_num}/{total_qs}] {question.question_id} (cat {question.category}: {cat_name})")
            print(f"    Q: {question.question[:80]}...")
            gold = question.answer or question.adversarial_answer
            print(f"    A: {gold[:80]}...")

            entry: dict = {
                "question_id": question.question_id,
                "conv_id": question.conv_id,
                "category": question.category,
                "category_name": cat_name,
                "question": question.question,
                "answer": question.answer,
                "adversarial_answer": question.adversarial_answer,
                "evidence": question.evidence,
            }

            # Baseline
            if not args.skip_baseline:
                try:
                    bl_result = run_baseline(
                        conv, question, budget,
                        model=args.baseline_model,
                        provider=args.baseline_provider,
                        cache_dir=cache_dir,
                    )
                    bl_result["f1"] = round(score_question(bl_result["hypothesis"], question), 4)
                    if args.judge:
                        try:
                            jr = judge_question(
                                bl_result["hypothesis"], question,
                                model=args.judge_model, provider=args.judge_provider,
                            )
                            bl_result["judge"] = jr["correct"]
                            bl_result["judge_reasoning"] = jr["reasoning"]
                        except Exception as je:
                            logger.warning("Judge failed for baseline %s: %s", question.question_id, je)
                            bl_result["judge"] = None
                    entry["baseline"] = bl_result
                    hyp = bl_result["hypothesis"][:100]
                    judge_str = ""
                    if args.judge and bl_result.get("judge") is not None:
                        judge_str = f"  judge={'CORRECT' if bl_result['judge'] else 'WRONG'}"
                    print(f"    Baseline: \"{hyp}...\"  F1={bl_result['f1']:.3f}{judge_str}  (${bl_result['cost']:.4f})")
                except Exception as e:
                    logger.error("Baseline failed for %s: %s", question.question_id, e)
                    entry["baseline"] = {"error": str(e)}
                    print(f"    Baseline: ERROR — {e}")

            # VC
            if not args.skip_vc and engine and messages:
                try:
                    vc_result = query_question(
                        engine, messages, question, conv, budget,
                        reader_model=args.reader_model,
                        reader_provider=args.reader_provider,
                        cache_dir=cache_dir,
                        curation_enabled=args.curation,
                        explain=args.explain,
                    )
                    vc_result["f1"] = round(score_question(vc_result["hypothesis"], question), 4)
                    if args.judge:
                        try:
                            jr = judge_question(
                                vc_result["hypothesis"], question,
                                model=args.judge_model, provider=args.judge_provider,
                            )
                            vc_result["judge"] = jr["correct"]
                            vc_result["judge_reasoning"] = jr["reasoning"]
                        except Exception as je:
                            logger.warning("Judge failed for vc %s: %s", question.question_id, je)
                            vc_result["judge"] = None
                    entry["vc"] = vc_result
                    hyp = vc_result["hypothesis"][:100]
                    tags = vc_result.get("tags_matched", [])[:5]
                    judge_str = ""
                    if args.judge and vc_result.get("judge") is not None:
                        judge_str = f"  judge={'CORRECT' if vc_result['judge'] else 'WRONG'}"
                    print(f"    VC: \"{hyp}...\"  F1={vc_result['f1']:.3f}{judge_str}")
                    print(f"        tags: {tags}, {vc_result.get('tokens_injected', 0)}t injected")
                except Exception as e:
                    logger.error("VC failed for %s: %s", question.question_id, e, exc_info=args.verbose)
                    entry["vc"] = {"error": str(e)}
                    print(f"    VC: ERROR — {e}")

            results.append(entry)
            _incremental_save(results, output_path, args, budget)

        # Close engine after all questions for this conversation
        if engine:
            engine.close()

    # Final summary
    summary = _compute_summary(results)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"\n  Total cost: ${budget.spent:.4f} / ${args.budget:.2f}")

    for method in ("baseline", "vc"):
        key = f"{method}_f1_mean"
        if key in summary:
            total_key = f"{method}_total"
            judge_key = f"{method}_judge_accuracy"
            judge_str = ""
            if judge_key in summary:
                judge_str = f"  Judge={summary[judge_key]:.4f}"
            print(f"  {method.upper()}: F1={summary[key]:.4f}{judge_str} ({summary[total_key]} questions)")
            per_cat = summary.get(f"{method}_per_category", {})
            for cat, info in sorted(per_cat.items()):
                marker = "+" if info["f1_mean"] >= 0.5 else "-"
                j_str = ""
                if "judge_accuracy" in info:
                    j_str = f"  Judge={info['judge_accuracy']:.4f}"
                print(f"    {marker} cat {cat} ({info['name']}): F1={info['f1_mean']:.4f}{j_str} ({info['count']}q)")

    print(f"{'=' * 70}\n")

    # Save final with summary
    final_data = {
        "benchmark": "locomo",
        "config": {
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "baseline_model": args.baseline_model,
            "baseline_provider": args.baseline_provider,
            "tagger_model": args.tagger_model,
            "tagger_provider": args.tagger_provider,
        },
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 6),
        "summary": summary,
        "questions": results,
    }
    output_path.write_text(json.dumps(final_data, indent=2, default=str))


if __name__ == "__main__":
    main()
