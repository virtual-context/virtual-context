"""CLI entry point for running the LongMemEval benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autopsy_report import write_autopsy_reports
from .baseline import run_baseline
from .cost import BudgetTracker
from .dataset import download_dataset, load_dataset, select_questions
from .judge import judge_answer
from .tally import refresh_running_tally
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
            "baseline_provider": args.baseline_provider,
            "baseline_auth_mode": args.baseline_auth_mode,
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "reader_auth_mode": args.reader_auth_mode,
            "tagger_provider": args.tagger_provider,
            "tagger_model": args.tagger_model,
            "summarizer_provider": args.summarizer_provider or args.tagger_provider,
            "summarizer_model": args.summarizer_model or args.tagger_model,
            "internal_openai_auth_mode": args.internal_openai_auth_mode,
            "reader_diagnostic_rationale": args.reader_diagnostic_rationale,
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_auth_mode": args.judge_auth_mode,
            "judge_votes": args.judge_votes,
            "judge_vote_mode": args.judge_vote_mode,
            "vc_on_baseline_wrong_only": args.vc_on_baseline_wrong_only,
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


def _judge_with_votes(
    *,
    question: str,
    answer: str,
    hypothesis: str,
    question_type: str,
    budget: BudgetTracker,
    label: str,
    model: str,
    provider: str,
    auth_mode: str,
    votes: int,
    vote_mode: str,
) -> dict[str, Any]:
    """Run judge multiple times and aggregate to a final verdict."""
    if votes < 1:
        raise ValueError(f"judge votes must be >= 1 (got {votes})")

    attempts: list[dict[str, Any]] = []
    yes_votes = 0
    for idx in range(votes):
        vote_label = label if votes == 1 else f"{label}:v{idx + 1}"
        result = judge_answer(
            question=question,
            answer=answer,
            hypothesis=hypothesis,
            question_type=question_type,
            budget=budget,
            label=vote_label,
            model=model,
            provider=provider,
            auth_mode=auth_mode,
        )
        attempts.append(result)
        if result.get("correct"):
            yes_votes += 1

    if vote_mode == "best-of":
        final_correct = yes_votes > 0
    elif vote_mode == "majority":
        final_correct = yes_votes >= (votes // 2 + 1)
    else:
        raise ValueError(f"unsupported judge vote mode: {vote_mode}")

    if final_correct:
        chosen = next((a for a in attempts if a.get("correct")), attempts[0])
    else:
        chosen = next((a for a in attempts if not a.get("correct")), attempts[0])

    total_cost = sum(float(a.get("cost", 0.0)) for a in attempts)
    return {
        "correct": final_correct,
        "explanation": str(chosen.get("explanation", "")).strip(),
        "cost": total_cost,
        "votes": attempts,
        "yes_votes": yes_votes,
        "total_votes": votes,
        "vote_mode": vote_mode,
    }


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
    parser.add_argument("--baseline-model", type=str, default=None, help="Baseline reader model (REQUIRED if running baseline)")
    parser.add_argument(
        "--baseline-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "openai-codex", "gemini", "gemini-cli", "gemini-oauth"],
        help="Provider for baseline run (REQUIRED if running baseline)",
    )
    parser.add_argument(
        "--baseline-auth-mode",
        type=str,
        default="auto",
        choices=["auto", "api-key", "oauth"],
        help="Auth mode for baseline provider (OAuth mode applies to OpenAI).",
    )
    parser.add_argument("--reader-model", type=str, default=None, help="VC reader model (REQUIRED if running VC)")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model (REQUIRED)")
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "openai-codex", "gemini", "gemini-cli", "gemini-oauth"],
        help="Provider for judge model (REQUIRED)",
    )
    parser.add_argument(
        "--judge-auth-mode",
        type=str,
        default="auto",
        choices=["auto", "api-key", "oauth"],
        help="Auth mode for judge provider (OAuth mode applies to OpenAI).",
    )
    parser.add_argument(
        "--judge-votes",
        type=int,
        default=1,
        help="Number of judge calls per answer (default: 1).",
    )
    parser.add_argument(
        "--judge-vote-mode",
        type=str,
        default="best-of",
        choices=["best-of", "majority"],
        help="How to aggregate judge votes (default: best-of).",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline, run VC only")
    parser.add_argument("--skip-vc", action="store_true", help="Skip VC, run baseline only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-completed questions.")
    parser.add_argument(
        "--vc-on-baseline-wrong-only",
        action="store_true",
        help=(
            "Run baseline first, then run VC only for questions where the baseline is judged WRONG. "
            "Requires baseline to be enabled."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--fresh", action="store_true", help="Clear cached ingestion+compaction (re-ingest from scratch)")
    parser.add_argument("--recompact", action="store_true", help="Keep cached tags, re-run compaction only (faster than --fresh)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all VC caches and exit")
    parser.add_argument("--summarizer-model", type=str, default=None, help="Override summarization model (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--tagger-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "ollama_native", "openrouter"],
        help="Provider for ingestion tag generation (REQUIRED if running VC)",
    )
    parser.add_argument(
        "--tagger-model",
        type=str,
        default=None,
        help="Model for ingestion tag generation (REQUIRED if running VC)",
    )
    parser.add_argument(
        "--summarizer-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "ollama_native", "openrouter"],
        help="Provider for compaction summarization (default: same as tagger-provider)",
    )
    parser.add_argument("--reader-provider", type=str, default=None,
                        choices=["anthropic", "openai", "openai-codex", "gemini"],
                        help="LLM provider for reader model (REQUIRED if running VC)")
    parser.add_argument(
        "--reader-auth-mode",
        type=str,
        default="auto",
        choices=["auto", "api-key", "oauth"],
        help=(
            "Reader auth mode. For OpenAI: auto=OAuth token first then OPENAI_API_KEY, "
            "api-key=OPENAI_API_KEY only, oauth=OAuth token only."
        ),
    )
    parser.add_argument(
        "--internal-openai-auth-mode",
        type=str,
        default="auto",
        choices=["auto", "api-key", "oauth"],
        help=(
            "Auth mode for internal OpenAI tagger/summarizer providers. "
            "auto=OAuth token first then OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--reader-diagnostic-rationale",
        action="store_true",
        help=(
            "Append a strict answer+evidence+conflict diagnostic format to the "
            "VC reader prompt for easier failure analysis."
        ),
    )
    parser.add_argument(
        "--openai-codex",
        action="store_true",
        help="Shortcut preset: reader/baseline/judge use openai-codex over OAuth",
    )
    parser.add_argument(
        "--openai-all-oauth",
        action="store_true",
        help=(
            "Shortcut preset for OAuth-only OpenAI pipeline: "
            "reader=gpt-5-codex, tagger=gpt-5-mini, summarizer=gpt-5-mini"
        ),
    )
    parser.add_argument(
        "--no-autopsy-report",
        action="store_true",
        help="Disable Autopsy report sidecar generation.",
    )
    parser.add_argument(
        "--autopsy-output-prefix",
        type=str,
        default=None,
        help="Optional output prefix for Autopsy report files (without extension).",
    )
    parser.add_argument("--download-only", action="store_true", help="Just download the dataset and exit")
    parser.add_argument(
        "--tagger-mode",
        type=str,
        default="combined",
        choices=["combined", "split"],
        help="Tag generation mode: 'combined' (single call) or 'split' (parallel tag+fact calls)",
    )
    parser.add_argument("--fact-provider", type=str, default=None, help="Provider for fact extraction in split mode")
    parser.add_argument("--fact-model", type=str, default=None, help="Model for fact extraction in split mode")
    parser.add_argument(
        "--parallel-ingest",
        type=int,
        default=1,
        help="Number of questions to ingest in parallel (default: 1)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run ingest + compact only, skip reader and judge phases. Save caches for later use.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override VC cache directory (default: benchmarks/longmemeval/cache/)",
    )
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

    args = parser.parse_args(argv)
    if args.judge_votes < 1:
        parser.error("--judge-votes must be >= 1")
    if args.judge_vote_mode == "majority" and args.judge_votes % 2 == 0:
        parser.error("--judge-votes must be odd when --judge-vote-mode=majority")
    if args.vc_on_baseline_wrong_only and args.skip_baseline:
        parser.error("--vc-on-baseline-wrong-only cannot be used with --skip-baseline")

    # Validate required model/provider args — NO silent defaults allowed.
    # Check MEMORY.md for model assignment rules before running.
    _NO_DEFAULT_MSG = (
        " No defaults — you must specify explicitly."
        " Check MEMORY.md for model assignment rules before running."
    )
    if not args.skip_baseline and not args.openai_codex and not args.openai_all_oauth:
        if not args.baseline_model or not args.baseline_provider:
            parser.error("--baseline-model and --baseline-provider are required." + _NO_DEFAULT_MSG)
    if not args.skip_vc and not args.openai_codex and not args.openai_all_oauth:
        if not args.reader_model or not args.reader_provider:
            parser.error("--reader-model and --reader-provider are required." + _NO_DEFAULT_MSG)
        if not args.tagger_provider or not args.tagger_model:
            parser.error("--tagger-provider and --tagger-model are required." + _NO_DEFAULT_MSG)
    if not args.judge_model or not args.judge_provider:
        if not args.openai_codex and not args.openai_all_oauth:
            parser.error("--judge-model and --judge-provider are required." + _NO_DEFAULT_MSG)

    _load_dotenv_if_present()
    if args.openai_codex:
        args.baseline_provider = "openai-codex"
        args.baseline_model = "gpt-5.3-codex"
        args.baseline_auth_mode = "oauth"
        args.reader_provider = "openai-codex"
        args.reader_model = "gpt-5.3-codex"
        args.reader_auth_mode = "oauth"
        args.judge_provider = "openai-codex"
        args.judge_model = "gpt-5.3-codex"
        args.judge_auth_mode = "oauth"
        args.tagger_provider = "openai"
        args.tagger_model = "gpt-5-mini"
        args.summarizer_provider = "openai"
        args.summarizer_model = "gpt-5-mini"
        args.internal_openai_auth_mode = "oauth"
    if args.openai_all_oauth:
        args.baseline_provider = "openai-codex"
        args.baseline_model = "gpt-5.3-codex"
        args.baseline_auth_mode = "oauth"
        args.reader_provider = "openai-codex"
        args.reader_model = "gpt-5.3-codex"
        args.reader_auth_mode = "oauth"
        args.tagger_provider = "openai"
        args.tagger_model = "gpt-5-mini"
        args.summarizer_provider = "openai"
        args.summarizer_model = "gpt-5-mini"
        args.internal_openai_auth_mode = "oauth"
        args.judge_provider = "openai-codex"
        args.judge_model = "gpt-5.3-codex"
        args.judge_auth_mode = "oauth"
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

    # Resume: load existing results and skip already-completed questions
    completed_qids: set[str] = set()
    if args.resume and args.output and Path(args.output).exists():
        import json as _json
        with open(args.output) as f:
            prev = _json.load(f)
        for entry in prev.get("questions", []):
            qid = entry.get("question_id", "")
            # Consider done if it has a non-error baseline or vc result
            bl = entry.get("baseline", {})
            vc = entry.get("vc", {})
            bl_done = isinstance(bl, dict) and "hypothesis" in bl and "error" not in bl
            vc_done = isinstance(vc, dict) and "hypothesis" in vc and "error" not in vc
            if (args.skip_vc and bl_done) or (args.skip_baseline and vc_done) or (bl_done and vc_done):
                completed_qids.add(qid)
                results.append(entry)
        if completed_qids:
            logger.info("Resume: loaded %d completed questions from %s", len(completed_qids), args.output)

    # Ingest-only mode: run ingestion + compaction, skip reader + judge
    if args.ingest_only:
        print(f"\n{'='*70}")
        print(f"LongMemEval INGEST-ONLY — {len(questions)} questions")
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
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                supersession=args.supersession,
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
                        print(
                            f"  [{i}/{len(questions)}] {qid}: "
                            f"{result['turns_ingested']} turns, "
                            f"{result['compaction_events']} compactions, "
                            f"{result['elapsed_s']}s"
                        )
                    except Exception as e:
                        logger.error("Ingest failed for %s: %s", q.question_id, e)
        else:
            for i, q in enumerate(questions, 1):
                try:
                    qid, result = _ingest_one(q)
                    total_turns += result.get("turns_ingested", 0)
                    total_compactions += result.get("compaction_events", 0)
                    print(
                        f"  [{i}/{len(questions)}] {qid}: "
                        f"{result['turns_ingested']} turns, "
                        f"{result['compaction_events']} compactions, "
                        f"{result['elapsed_s']}s"
                    )
                except Exception as e:
                    logger.error("Ingest failed for %s: %s", q.question_id, e)

        elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"Ingest complete: {len(questions)} questions, {total_turns} turns, "
              f"{total_compactions} compactions in {elapsed:.0f}s")
        print(f"Caches saved to: {Path(__file__).parent / 'cache'}")
        print(f"{'='*70}\n")
        return

    print(f"\n{'='*70}")
    print(f"LongMemEval Smoke Test — {len(questions)} questions, ${args.budget:.2f} budget")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        if q.question_id in completed_qids:
            print(f"\n--- Question {i}/{len(questions)}: {q.question_id} — RESUMED (skip) ---")
            continue
        print(f"\n--- Question {i}/{len(questions)}: {q.question_id} ({q.question_type}) ---")
        print(f"  Q: {q.question[:120]}...")
        print(f"  A: {q.answer[:120]}...")
        print(f"  Haystack: ~{q.haystack_tokens_est:,} tokens, {len(q.haystack_sessions)} sessions")

        # Budget check (disabled — cost tracking is inaccurate, see BUG-030)
        # est_cost = budget.estimate_question_cost(q.haystack_tokens_est)
        # if not budget.can_afford(est_cost):
        #     print(f"  SKIP: estimated cost ${est_cost:.2f} exceeds remaining budget ${budget.remaining:.2f}")
        #     break

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
                print(f"  Running baseline ({args.baseline_provider}/{args.baseline_model})...")
                t0 = time.time()
                baseline_result = run_baseline(
                    q,
                    budget,
                    model=args.baseline_model,
                    provider=args.baseline_provider,
                    auth_mode=args.baseline_auth_mode,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                )
                baseline_result["elapsed_s"] = round(time.time() - t0, 1)
                print(f"  Baseline: \"{baseline_result['hypothesis'][:100]}...\"  (${baseline_result['cost']:.4f})")

                # Judge baseline (multi-vote aggregation)
                baseline_judge = _judge_with_votes(
                    question=q.question,
                    answer=q.answer,
                    hypothesis=baseline_result["hypothesis"],
                    question_type=q.question_type,
                    budget=budget,
                    label=f"baseline:{q.question_id}",
                    model=args.judge_model,
                    provider=args.judge_provider,
                    auth_mode=args.judge_auth_mode,
                    votes=args.judge_votes,
                    vote_mode=args.judge_vote_mode,
                )
                baseline_result["correct"] = baseline_judge["correct"]
                baseline_result["judge_explanation"] = baseline_judge["explanation"]
                baseline_result["judge_votes"] = baseline_judge["votes"]
                baseline_result["judge_yes_votes"] = baseline_judge["yes_votes"]
                baseline_result["judge_total_votes"] = baseline_judge["total_votes"]
                baseline_result["judge_vote_mode"] = baseline_judge["vote_mode"]
                baseline_result["judge_cost"] = baseline_judge["cost"]
                print(
                    f"  Baseline judge: {'CORRECT' if baseline_judge['correct'] else 'WRONG'} "
                    f"({baseline_judge['yes_votes']}/{baseline_judge['total_votes']} yes, "
                    f"{baseline_judge['vote_mode']})"
                )

                entry["baseline"] = baseline_result
            except Exception as e:
                logger.error("Baseline failed for %s: %s", q.question_id, e)
                entry["baseline"] = {"error": str(e)}

        # Run VC (optionally gated on baseline being wrong)
        should_run_vc = not args.skip_vc
        if should_run_vc and args.vc_on_baseline_wrong_only:
            baseline_entry = entry.get("baseline")
            if not isinstance(baseline_entry, dict):
                should_run_vc = False
                entry["vc_skipped_reason"] = "baseline_missing"
                print("  VC skipped: baseline result missing (gated mode enabled)")
            elif "error" in baseline_entry:
                should_run_vc = False
                entry["vc_skipped_reason"] = "baseline_error"
                print("  VC skipped: baseline failed (gated mode enabled)")
            elif baseline_entry.get("correct") is True:
                should_run_vc = False
                entry["vc_skipped_reason"] = "baseline_correct"
                print("  VC skipped: baseline already CORRECT (gated mode enabled)")
            elif baseline_entry.get("correct") is False:
                print("  VC target: baseline WRONG, running VC...")
            else:
                should_run_vc = False
                entry["vc_skipped_reason"] = "baseline_unjudged"
                print("  VC skipped: baseline not judged (gated mode enabled)")

        if should_run_vc:
            try:
                print(f"  Running VC (window={args.context_window})...")
                t0 = time.time()
                cache_dir_path = Path(args.cache_dir) if args.cache_dir else None
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
                    reader_diagnostic_rationale=args.reader_diagnostic_rationale,
                    tagger_mode=args.tagger_mode,
                    fact_provider=args.fact_provider,
                    fact_model=args.fact_model,
                    cache_dir=cache_dir_path,
                    curation_enabled=args.curation,
                    curation_provider=args.curation_provider,
                    curation_model=args.curation_model,
                    supersession=args.supersession,
                )
                vc_result["elapsed_s"] = round(time.time() - t0, 1)
                print(f"  VC: \"{vc_result['hypothesis'][:100]}...\"  (${vc_result['cost']:.4f})")
                print(
                    f"      tags: {vc_result['tags_matched'][:5]}, "
                    f"{vc_result['tokens_injected']}t prompt(est), "
                    f"{vc_result['compaction_events']} compactions"
                )

                # Judge VC (multi-vote aggregation)
                vc_judge = _judge_with_votes(
                    question=q.question,
                    answer=q.answer,
                    hypothesis=vc_result["hypothesis"],
                    question_type=q.question_type,
                    budget=budget,
                    label=f"vc:{q.question_id}",
                    model=args.judge_model,
                    provider=args.judge_provider,
                    auth_mode=args.judge_auth_mode,
                    votes=args.judge_votes,
                    vote_mode=args.judge_vote_mode,
                )
                vc_result["correct"] = vc_judge["correct"]
                vc_result["judge_explanation"] = vc_judge["explanation"]
                vc_result["judge_votes"] = vc_judge["votes"]
                vc_result["judge_yes_votes"] = vc_judge["yes_votes"]
                vc_result["judge_total_votes"] = vc_judge["total_votes"]
                vc_result["judge_vote_mode"] = vc_judge["vote_mode"]
                vc_result["judge_cost"] = vc_judge["cost"]
                print(
                    f"  VC judge: {'CORRECT' if vc_judge['correct'] else 'WRONG'} "
                    f"({vc_judge['yes_votes']}/{vc_judge['total_votes']} yes, "
                    f"{vc_judge['vote_mode']})"
                )

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
            "baseline_provider": args.baseline_provider,
            "baseline_auth_mode": args.baseline_auth_mode,
            "reader_model": args.reader_model,
            "reader_provider": args.reader_provider,
            "reader_auth_mode": args.reader_auth_mode,
            "tagger_provider": args.tagger_provider,
            "tagger_model": args.tagger_model,
            "summarizer_provider": args.summarizer_provider or args.tagger_provider,
            "summarizer_model": args.summarizer_model or args.tagger_model,
            "internal_openai_auth_mode": args.internal_openai_auth_mode,
            "reader_diagnostic_rationale": args.reader_diagnostic_rationale,
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_auth_mode": args.judge_auth_mode,
            "judge_votes": args.judge_votes,
            "judge_vote_mode": args.judge_vote_mode,
            "vc_on_baseline_wrong_only": args.vc_on_baseline_wrong_only,
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
    if not args.no_autopsy_report:
        autopsy_json, autopsy_md = write_autopsy_reports(
            results_data=output_data,
            results_output_path=out_path,
            autopsy_output_prefix=args.autopsy_output_prefix,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        print(f"Autopsy report (JSON): {autopsy_json}")
        print(f"Autopsy report (Markdown): {autopsy_md}")
    try:
        tally_path = refresh_running_tally()
        print(f"Running tally: {tally_path}")
    except Exception as e:
        logger.warning("Failed to refresh running tally: %s", e)
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
