"""CLI entry point for running the MRCR benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Force unbuffered output so background runs stream progress
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.longmemeval.cost import BudgetTracker

from .autopsy_report import write_autopsy_reports
from .baseline import run_baseline
from .dataset import download_dataset, load_dataset, select_questions
from .grader import grade
from .vc_runner import clear_cache, run_vc, run_vc_ingest_only

logger = logging.getLogger(__name__)


def _load_dotenv_if_present() -> None:
    """Load KEY=VALUE pairs from repo-local .env without overriding existing env."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    loaded = 0
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if ((value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        logger.info("Loaded %d env var(s) from %s", loaded, env_path)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _incremental_save(results: list[dict], args, budget) -> None:
    """Write results after each question so progress survives process kills."""
    if args.output is None:
        args.output = f"mrcr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = Path(args.output)
    data = {
        "benchmark": "mrcr",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "config": _build_config_dict(args),
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 4),
        "questions": results,
        "partial": True,
    }
    out_path.write_text(json.dumps(data, indent=2, default=str))


def _build_config_dict(args) -> dict:
    return {
        "context_window": args.context_window,
        "baseline_model": args.baseline_model,
        "baseline_provider": args.baseline_provider,
        "baseline_auth_mode": args.baseline_auth_mode,
        "reader_model": args.reader_model,
        "reader_provider": args.reader_provider,
        "reader_auth_mode": args.reader_auth_mode,
        "tagger_provider": args.tagger_provider,
        "tagger_model": args.tagger_model,
        "summarizer_provider": args.summarizer_provider or (args.tagger_provider if hasattr(args, "tagger_provider") else None),
        "summarizer_model": args.summarizer_model or (args.tagger_model if hasattr(args, "tagger_model") else None),
    }


def _compute_summary(results: list[dict], args: argparse.Namespace) -> dict:
    """Compute score summary from results."""
    summary: dict[str, Any] = {}

    for method in ["baseline", "vc"]:
        if (method == "baseline" and args.skip_baseline) or (method == "vc" and args.skip_vc):
            continue

        scores: list[float] = []
        per_needle: dict[int, list[float]] = {}
        per_bin: dict[str, list[float]] = {}

        for r in results:
            if method not in r or "error" in r.get(method, {}):
                continue
            score = r[method].get("score", 0.0)
            scores.append(score)

            n = r.get("n_needles", 0)
            per_needle.setdefault(n, []).append(score)

            b = r.get("context_bin", "")
            per_bin.setdefault(b, []).append(score)

        avg = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for s in scores if s >= 0.8)

        summary[f"{method}_avg_score"] = round(avg, 4)
        summary[f"{method}_pass_count"] = pass_count
        summary[f"{method}_total"] = len(scores)

    # Cross-method per-needle and per-bin breakdowns
    per_needle_combined: dict[str, dict] = {}
    per_bin_combined: dict[str, dict] = {}

    for r in results:
        n_key = str(r.get("n_needles", "?"))
        bin_key = r.get("context_bin", "?")

        for method in ["baseline", "vc"]:
            if method not in r or "error" in r.get(method, {}):
                continue
            score = r[method].get("score", 0.0)

            if n_key not in per_needle_combined:
                per_needle_combined[n_key] = {"count": 0}
            per_needle_combined[n_key]["count"] += 1
            per_needle_combined[n_key].setdefault(f"{method}_scores", []).append(score)

            if bin_key not in per_bin_combined:
                per_bin_combined[bin_key] = {"count": 0}
            per_bin_combined[bin_key].setdefault(f"{method}_scores", []).append(score)

    # Compute averages
    for n_key, data in per_needle_combined.items():
        for method in ["baseline", "vc"]:
            scores_key = f"{method}_scores"
            if scores_key in data:
                s = data[scores_key]
                data[f"{method}_avg"] = round(sum(s) / len(s), 4) if s else 0.0
                data["count"] = max(data["count"], len(s))
                del data[scores_key]

    for bin_key, data in per_bin_combined.items():
        for method in ["baseline", "vc"]:
            scores_key = f"{method}_scores"
            if scores_key in data:
                s = data[scores_key]
                data[f"{method}_avg"] = round(sum(s) / len(s), 4) if s else 0.0
                data["count"] = max(data["count"], len(s))
                del data[scores_key]

    summary["per_needle"] = per_needle_combined
    summary["per_bin"] = per_bin_combined

    return summary


def _print_summary(summary: dict, budget: BudgetTracker) -> None:
    print(f"\n  Total cost: ${budget.spent:.4f} / ${budget.budget:.2f}")

    for method in ["baseline", "vc"]:
        avg_key = f"{method}_avg_score"
        if avg_key in summary:
            avg = summary[avg_key]
            total = summary.get(f"{method}_total", 0)
            pass_count = summary.get(f"{method}_pass_count", 0)
            print(f"  {method.upper():>10}: avg={avg:.3f}, pass={pass_count}/{total} (>= 0.8)")

    per_needle = summary.get("per_needle", {})
    if per_needle:
        print("\n  Per needle count:")
        for n_key, data in sorted(per_needle.items()):
            parts = [f"    {n_key}n:"]
            for method in ["baseline", "vc"]:
                avg = data.get(f"{method}_avg")
                if avg is not None:
                    parts.append(f"{method}={avg:.3f}")
            parts.append(f"(n={data.get('count', 0)})")
            print(" ".join(parts))

    per_bin = summary.get("per_bin", {})
    if per_bin:
        print("\n  Per context bin:")
        for bin_key, data in sorted(per_bin.items()):
            parts = [f"    {bin_key}:"]
            for method in ["baseline", "vc"]:
                avg = data.get(f"{method}_avg")
                if avg is not None:
                    parts.append(f"{method}={avg:.3f}")
            parts.append(f"(n={data.get('count', 0)})")
            print(" ".join(parts))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MRCR benchmark: compare VC vs full-context baseline",
        prog="python -m benchmarks.mrcr",
    )
    parser.add_argument("--budget", type=float, default=5.0, help="Max spend in USD (default: $5)")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to run (default: 5)")
    parser.add_argument("--questions", nargs="+", help="Specific question IDs to run")
    parser.add_argument("--needles", type=int, nargs="+", choices=[2, 4, 8], help="Filter by needle count")
    parser.add_argument("--bins", nargs="+", help="Filter by context bin (e.g. 128k-256k, 256k-512k)")
    parser.add_argument("--context-window", type=int, default=65536, help="VC context window (default: 65536)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    # Model/provider args
    parser.add_argument("--baseline-model", type=str, default=None, help="Baseline model (REQUIRED if running baseline)")
    parser.add_argument(
        "--baseline-provider", type=str, default=None,
        choices=["anthropic", "openai", "openai-codex", "gemini"],
        help="Provider for baseline (REQUIRED if running baseline)",
    )
    parser.add_argument("--baseline-auth-mode", type=str, default="auto", choices=["auto", "api-key", "oauth"])
    parser.add_argument("--reader-model", type=str, default=None, help="VC reader model (REQUIRED if running VC)")
    parser.add_argument(
        "--reader-provider", type=str, default=None,
        choices=["anthropic", "openai", "openai-responses", "openai-codex", "openrouter", "gemini"],
        help="Provider for VC reader (REQUIRED if running VC)",
    )
    parser.add_argument("--reader-auth-mode", type=str, default="auto", choices=["auto", "api-key", "oauth"])
    parser.add_argument(
        "--tagger-provider", type=str, default=None,
        choices=["anthropic", "openai", "ollama_native", "openrouter"],
        help="Provider for tag generation (REQUIRED if running VC)",
    )
    parser.add_argument("--tagger-model", type=str, default=None, help="Model for tag generation (REQUIRED if running VC)")
    parser.add_argument(
        "--summarizer-provider", type=str, default=None,
        choices=["anthropic", "openai", "ollama_native", "openrouter"],
        help="Provider for compaction summarization (default: tagger-provider)",
    )
    parser.add_argument("--summarizer-model", type=str, default=None, help="Override summarization model")
    parser.add_argument(
        "--internal-openai-auth-mode", type=str, default="auto",
        choices=["auto", "api-key", "oauth"],
    )

    # Control flags
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline, run VC only")
    parser.add_argument("--skip-vc", action="store_true", help="Skip VC, run baseline only")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--fresh-never-use-this-except-with-permission", dest="fresh",
        action="store_true", help="DANGER: Clear cached ingestion+compaction",
    )
    parser.add_argument("--recompact", action="store_true", help="Keep tags, re-run compaction only")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all VC caches and exit")
    parser.add_argument("--download-only", action="store_true", help="Just download the dataset and exit")
    parser.add_argument("--no-autopsy-report", action="store_true", help="Disable autopsy report")
    parser.add_argument("--autopsy-output-prefix", type=str, default=None)
    parser.add_argument(
        "--tagger-mode", type=str, default="combined",
        choices=["combined", "split"],
    )
    parser.add_argument(
        "--parallel-ingest", type=int, default=1,
        help="Number of questions to ingest in parallel (default: 1)",
    )
    parser.add_argument("--ingest-only", action="store_true", help="Run ingest + compact only")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override VC cache directory")
    parser.add_argument("--verbose-reasoning", action="store_true", default=False)
    parser.add_argument("--fact-provider", type=str, default=None, help="Provider for fact extraction in split mode")
    parser.add_argument("--fact-model", type=str, default=None, help="Model for fact extraction in split mode")
    parser.add_argument(
        "--curation",
        action="store_true",
        default=False,
        help="Enable LLM-based fact curation",
    )
    parser.add_argument("--curation-provider", type=str, default=None, help="Provider for fact curation (default: summarizer provider)")
    parser.add_argument("--curation-model", type=str, default=None, help="Model for fact curation (default: summarizer model)")
    parser.add_argument(
        "--supersession",
        action="store_true",
        default=False,
        help="Run fact supersession pass after compaction (deduplicates facts)",
    )
    parser.add_argument("--supersession-provider", type=str, default=None, help="Provider for supersession (default: fact provider, then summarizer)")
    parser.add_argument("--supersession-model", type=str, default=None, help="Model for supersession (default: fact model, then summarizer)")

    args = parser.parse_args(argv)

    # Validate required args
    _NO_DEFAULT_MSG = " No defaults — you must specify explicitly."
    if not args.skip_baseline and not args.download_only and not args.clear_cache and not args.ingest_only:
        if not args.baseline_model or not args.baseline_provider:
            parser.error("--baseline-model and --baseline-provider are required." + _NO_DEFAULT_MSG)
    if not args.skip_vc and not args.download_only and not args.clear_cache:
        if not args.reader_model or not args.reader_provider:
            if not args.ingest_only:
                parser.error("--reader-model and --reader-provider are required." + _NO_DEFAULT_MSG)
        if not args.tagger_provider or not args.tagger_model:
            parser.error("--tagger-provider and --tagger-model are required." + _NO_DEFAULT_MSG)

    _load_dotenv_if_present()
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
    dataset = load_dataset(needle_counts=args.needles)
    questions = select_questions(
        dataset,
        count=args.count,
        needle_counts=args.needles,
        bins=args.bins,
        question_ids=args.questions,
    )

    if not questions:
        print("No questions selected. Check your filters.")
        return

    budget = BudgetTracker(budget=args.budget)
    results: list[dict] = []

    # Resume
    completed_qids: set[str] = set()
    if args.resume and args.output and Path(args.output).exists():
        with open(args.output) as f:
            prev = json.load(f)
        for entry in prev.get("questions", []):
            qid = entry.get("question_id", "")
            bl = entry.get("baseline", {})
            vc = entry.get("vc", {})
            bl_done = isinstance(bl, dict) and "hypothesis" in bl and "error" not in bl
            vc_done = isinstance(vc, dict) and "hypothesis" in vc and "error" not in vc
            if (args.skip_vc and bl_done) or (args.skip_baseline and vc_done) or (bl_done and vc_done):
                completed_qids.add(qid)
                results.append(entry)
        if completed_qids:
            logger.info("Resume: loaded %d completed questions from %s", len(completed_qids), args.output)

    # Ingest-only mode
    if args.ingest_only:
        print(f"\n{'='*70}")
        print(f"MRCR INGEST-ONLY — {len(questions)} questions")
        print(f"{'='*70}\n")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _ingest_one(q):
            t0 = time.time()
            result = run_vc_ingest_only(
                q,
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
                curation_enabled=args.curation,
                curation_provider=args.curation_provider,
                curation_model=args.curation_model,
                supersession=args.supersession,
                supersession_provider=args.supersession_provider,
                supersession_model=args.supersession_model,
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            )
            result["elapsed_s"] = round(time.time() - t0, 1)
            return q.question_id, result

        n_workers = max(1, args.parallel_ingest)
        total_turns = 0
        total_compactions = 0
        t_start = time.time()

        if n_workers > 1:
            print(f"  Parallel ingestion: {n_workers} workers\n")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_ingest_one, q): q for q in questions}
                for i, future in enumerate(as_completed(futures), 1):
                    q = futures[future]
                    try:
                        qid, result = future.result()
                        total_turns += result.get("turns_ingested", 0)
                        total_compactions += result.get("compaction_events", 0)
                        print(f"  [{i}/{len(questions)}] {qid}: {result['turns_ingested']} turns, "
                              f"{result['compaction_events']} compactions, {result['elapsed_s']}s")
                    except Exception as e:
                        logger.error("Ingest failed for %s: %s", q.question_id, e)
        else:
            for i, q in enumerate(questions, 1):
                try:
                    qid, result = _ingest_one(q)
                    total_turns += result.get("turns_ingested", 0)
                    total_compactions += result.get("compaction_events", 0)
                    print(f"  [{i}/{len(questions)}] {qid}: {result['turns_ingested']} turns, "
                          f"{result['compaction_events']} compactions, {result['elapsed_s']}s")
                except Exception as e:
                    logger.error("Ingest failed for %s: %s", q.question_id, e)

        elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"Ingest complete: {len(questions)} questions, {total_turns} turns, "
              f"{total_compactions} compactions in {elapsed:.0f}s")
        print(f"{'='*70}\n")
        return

    # Main benchmark loop
    print(f"\n{'='*70}")
    print(f"MRCR Benchmark — {len(questions)} questions, ${args.budget:.2f} budget")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        if q.question_id in completed_qids:
            print(f"\n--- Question {i}/{len(questions)}: {q.question_id} — RESUMED (skip) ---")
            continue

        print(f"\n--- Question {i}/{len(questions)}: {q.question_id} ({q.n_needles}n, {q.context_bin}) ---")
        print(f"  Q: {q.question_message[:120]}...")
        print(f"  A: {q.answer[:80]}...")
        print(f"  Context: ~{q.tokens_est:,} tokens, {q.total_messages} messages")

        entry: dict = {
            "question_id": q.question_id,
            "n_needles": q.n_needles,
            "context_bin": q.context_bin,
            "question_preview": q.question_message[:200],
            "answer": q.answer[:500],
            "tokens_est": q.tokens_est,
            "total_messages": q.total_messages,
            "random_string": q.random_string,
        }

        # Run baseline
        if not args.skip_baseline:
            try:
                print(f"  Running baseline ({args.baseline_provider}/{args.baseline_model})...")
                t0 = time.time()
                baseline_result = run_baseline(
                    q, budget,
                    model=args.baseline_model,
                    provider=args.baseline_provider,
                    auth_mode=args.baseline_auth_mode,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                )
                baseline_result["elapsed_s"] = round(time.time() - t0, 1)

                # Grade
                baseline_score = grade(
                    baseline_result["hypothesis"],
                    q.answer,
                    q.random_string,
                )
                baseline_result["score"] = round(baseline_score, 4)

                print(f"  Baseline: score={baseline_score:.3f}, "
                      f"\"{baseline_result['hypothesis'][:80]}...\" (${baseline_result['cost']:.4f})")

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
                    reader_auth_mode=args.reader_auth_mode,
                    tagger_provider=args.tagger_provider,
                    tagger_model=args.tagger_model,
                    summarizer_provider=args.summarizer_provider,
                    internal_openai_auth_mode=args.internal_openai_auth_mode,
                    tagger_mode=args.tagger_mode,
                    fact_provider=args.fact_provider,
                    fact_model=args.fact_model,
                    curation_enabled=args.curation,
                    curation_provider=args.curation_provider,
                    curation_model=args.curation_model,
                    supersession=args.supersession,
                    supersession_provider=args.supersession_provider,
                    supersession_model=args.supersession_model,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                    verbose_reasoning=args.verbose_reasoning,
                )
                vc_result["elapsed_s"] = round(time.time() - t0, 1)

                # Grade
                vc_score = grade(
                    vc_result["hypothesis"],
                    q.answer,
                    q.random_string,
                )
                vc_result["score"] = round(vc_score, 4)

                print(f"  VC: score={vc_score:.3f}, "
                      f"\"{vc_result['hypothesis'][:80]}...\" (${vc_result['cost']:.4f})")
                print(f"      tags: {vc_result['tags_matched'][:5]}, "
                      f"{vc_result['tokens_injected']}t injected, "
                      f"{vc_result['compaction_events']} compactions")

                entry["vc"] = vc_result
            except Exception as e:
                logger.error("VC failed for %s: %s", q.question_id, e, exc_info=args.verbose)
                entry["vc"] = {"error": str(e)}

        results.append(entry)
        _incremental_save(results, args, budget)
        print(f"  Budget: ${budget.spent:.4f} / ${budget.budget:.2f} spent")

    # Compute summary
    summary = _compute_summary(results, args)

    output_data = {
        "benchmark": "mrcr",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "config": _build_config_dict(args),
        "budget_limit": args.budget,
        "actual_cost": round(budget.spent, 4),
        "questions": results,
        "summary": summary,
        "cost_breakdown": budget.summary(),
    }

    if args.output is None:
        args.output = f"mrcr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output_data, indent=2, default=str))

    print(f"\n{'='*70}")
    print(f"Results saved to: {out_path}")
    if not args.no_autopsy_report:
        autopsy_json, autopsy_md = write_autopsy_reports(
            results_data=output_data,
            results_output_path=out_path,
            autopsy_output_prefix=args.autopsy_output_prefix,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        print(f"Autopsy report (JSON): {autopsy_json}")
        print(f"Autopsy report (Markdown): {autopsy_md}")
    _print_summary(summary, budget)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
