"""Autopsy report generation for MRCR benchmark outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _find_latest(directory: Path, prefix: str) -> Path | None:
    """Find the newest timestamped file matching prefix_*.json."""
    candidates = sorted(directory.glob(f"{prefix}_*.json"))
    if candidates:
        return candidates[-1]
    legacy = directory / f"{prefix}.json"
    return legacy if legacy.exists() else None


def build_autopsy_data(results_data: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    questions = results_data.get("questions", [])

    baseline_total_cost = 0.0
    vc_total_cost = 0.0
    baseline_total_input = 0.0
    baseline_total_output = 0.0
    vc_total_input = 0.0
    vc_total_output = 0.0

    question_reports: list[dict[str, Any]] = []
    for q in questions:
        baseline = q.get("baseline", {}) if isinstance(q.get("baseline"), dict) else {}
        vc = q.get("vc", {}) if isinstance(q.get("vc"), dict) else {}

        baseline_input = _safe_num(baseline.get("input_tokens"))
        baseline_output = _safe_num(baseline.get("output_tokens"))
        baseline_cost = _safe_num(baseline.get("cost"))

        vc_input = _safe_num(vc.get("input_tokens"))
        vc_output = _safe_num(vc.get("output_tokens"))
        vc_cost = _safe_num(vc.get("cost"))
        vc_injected = _safe_num(vc.get("tokens_injected"))

        baseline_total_cost += baseline_cost
        vc_total_cost += vc_cost
        baseline_total_input += baseline_input
        baseline_total_output += baseline_output
        vc_total_input += vc_input
        vc_total_output += vc_output

        tool_calls = vc.get("tool_calls", []) if isinstance(vc.get("tool_calls"), list) else []

        qid = q.get("question_id", "")
        base_cache = cache_dir or Path("benchmarks/mrcr/cache")
        q_dir = base_cache / qid
        vc_log = _find_latest(q_dir, "payload_log") if q_dir.exists() else None
        bl_log = _find_latest(q_dir, "baseline_payload_log") if q_dir.exists() else None

        question_reports.append({
            "question_id": qid,
            "n_needles": q.get("n_needles"),
            "context_bin": q.get("context_bin"),
            "question_preview": q.get("question_preview", ""),
            "gold_answer_preview": (q.get("answer", "") or "")[:200],
            "payload_paths": {
                "vc": str(vc_log) if vc_log else "",
                "baseline": str(bl_log) if bl_log else "",
            },
            "baseline": {
                "answer_preview": (baseline.get("hypothesis", "") or "")[:200],
                "score": _safe_num(baseline.get("score")),
                "token_usage": {
                    "input_tokens": baseline_input,
                    "output_tokens": baseline_output,
                },
                "cost_usd": baseline_cost,
                "elapsed_s": _safe_num(baseline.get("elapsed_s")),
                "error": baseline.get("error"),
            },
            "vc": {
                "answer_preview": (vc.get("hypothesis", "") or "")[:200],
                "score": _safe_num(vc.get("score")),
                "token_usage": {
                    "reader_input_tokens": vc_input,
                    "reader_output_tokens": vc_output,
                    "tokens_injected": vc_injected,
                },
                "cost_usd": vc_cost,
                "elapsed_s": _safe_num(vc.get("elapsed_s")),
                "compaction_events": vc.get("compaction_events"),
                "tags_matched": vc.get("tags_matched", []),
                "tool_calls_count": len(tool_calls),
                "error": vc.get("error"),
            },
        })

    return {
        "report_type": "mrcr_autopsy_report",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "mrcr",
        "run_date": results_data.get("date"),
        "config": results_data.get("config", {}),
        "summary": results_data.get("summary", {}),
        "total_run_cost_usd": _safe_num(results_data.get("actual_cost")),
        "cost_comparison": {
            "baseline_total_cost_usd": baseline_total_cost,
            "vc_total_cost_usd": vc_total_cost,
            "delta_vc_minus_baseline_usd": vc_total_cost - baseline_total_cost,
        },
        "token_comparison": {
            "baseline_total_input_tokens": baseline_total_input,
            "baseline_total_output_tokens": baseline_total_output,
            "vc_total_reader_input_tokens": vc_total_input,
            "vc_total_reader_output_tokens": vc_total_output,
        },
        "questions": question_reports,
    }


def render_autopsy_markdown(autopsy: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MRCR Autopsy Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {autopsy.get('generated_at_utc', '')}")
    lines.append(f"- Run date: {autopsy.get('run_date', '')}")
    lines.append(f"- Total run cost: ${_safe_num(autopsy.get('total_run_cost_usd')):.4f}")

    cc = autopsy.get("cost_comparison", {})
    lines.append(f"- Baseline total cost: ${_safe_num(cc.get('baseline_total_cost_usd')):.4f}")
    lines.append(f"- VC total cost: ${_safe_num(cc.get('vc_total_cost_usd')):.4f}")
    lines.append("")

    summary = autopsy.get("summary", {})
    for method in ["baseline", "vc"]:
        avg_key = f"{method}_avg_score"
        if avg_key in summary:
            lines.append(f"- {method.upper()} avg score: {summary[avg_key]:.3f}")
    lines.append("")

    # Per-needle breakdown
    per_needle = summary.get("per_needle", {})
    if per_needle:
        lines.append("## Score by Needle Count")
        lines.append("")
        lines.append("| Needles | Baseline Avg | VC Avg | Count |")
        lines.append("|---------|-------------|--------|-------|")
        for needle_key, data in sorted(per_needle.items()):
            bl_avg = data.get("baseline_avg", 0)
            vc_avg = data.get("vc_avg", 0)
            cnt = data.get("count", 0)
            lines.append(f"| {needle_key} | {bl_avg:.3f} | {vc_avg:.3f} | {cnt} |")
        lines.append("")

    # Per-bin breakdown
    per_bin = summary.get("per_bin", {})
    if per_bin:
        lines.append("## Score by Context Bin")
        lines.append("")
        lines.append("| Bin | Baseline Avg | VC Avg | Count |")
        lines.append("|-----|-------------|--------|-------|")
        for bin_key, data in sorted(per_bin.items()):
            bl_avg = data.get("baseline_avg", 0)
            vc_avg = data.get("vc_avg", 0)
            cnt = data.get("count", 0)
            lines.append(f"| {bin_key} | {bl_avg:.3f} | {vc_avg:.3f} | {cnt} |")
        lines.append("")

    for q in autopsy.get("questions", []):
        lines.append(f"## {q.get('question_id', '')} ({q.get('n_needles')}n, {q.get('context_bin', '')})")
        lines.append("")

        bl = q.get("baseline", {})
        lines.append(f"### Baseline (score: {_safe_num(bl.get('score')):.3f})")
        lines.append(f"- Answer preview: {bl.get('answer_preview', '')}")
        lines.append(f"- Tokens: in={int(_safe_num(bl.get('token_usage', {}).get('input_tokens')))}, out={int(_safe_num(bl.get('token_usage', {}).get('output_tokens')))}")
        lines.append(f"- Cost: ${_safe_num(bl.get('cost_usd')):.4f}, Elapsed: {_safe_num(bl.get('elapsed_s')):.1f}s")
        if bl.get("error"):
            lines.append(f"- Error: {bl['error']}")
        lines.append("")

        vc = q.get("vc", {})
        lines.append(f"### VC (score: {_safe_num(vc.get('score')):.3f})")
        lines.append(f"- Answer preview: {vc.get('answer_preview', '')}")
        vt = vc.get("token_usage", {})
        lines.append(f"- Reader tokens: in={int(_safe_num(vt.get('reader_input_tokens')))}, out={int(_safe_num(vt.get('reader_output_tokens')))}")
        lines.append(f"- Injected: {int(_safe_num(vt.get('tokens_injected')))} tokens")
        lines.append(f"- Cost: ${_safe_num(vc.get('cost_usd')):.4f}, Elapsed: {_safe_num(vc.get('elapsed_s')):.1f}s")
        lines.append(f"- Tags matched: {', '.join(vc.get('tags_matched', [])[:10])}")
        lines.append(f"- Tool calls: {vc.get('tool_calls_count', 0)}, Compaction events: {vc.get('compaction_events', 0)}")
        if vc.get("error"):
            lines.append(f"- Error: {vc['error']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_autopsy_reports(
    *,
    results_data: dict[str, Any],
    results_output_path: Path,
    autopsy_output_prefix: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write Autopsy report JSON and Markdown sidecars."""
    autopsy = build_autopsy_data(results_data, cache_dir=cache_dir)

    if autopsy_output_prefix:
        prefix = Path(autopsy_output_prefix)
    else:
        prefix = results_output_path.with_suffix("")

    json_path = prefix.with_name(prefix.name + ".autopsy.json")
    md_path = prefix.with_name(prefix.name + ".autopsy.md")

    json_path.write_text(json.dumps(autopsy, indent=2, default=str))
    md_path.write_text(render_autopsy_markdown(autopsy))
    return json_path, md_path
