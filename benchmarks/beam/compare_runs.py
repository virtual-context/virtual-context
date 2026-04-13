from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _question_map(run_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    conversations = run_data.get("conversations", [])
    questions: dict[str, dict[str, Any]] = {}
    for conv in conversations:
        for question in conv.get("questions", []):
            questions[question["question_id"]] = question
    return questions


def _score(question: dict[str, Any]) -> float | None:
    judgment = question.get("judgment") or {}
    return judgment.get("score")


def _candidate_payload_paths(payload_dir: Path, question_id: str) -> list[Path]:
    raw = sorted(payload_dir.glob(f"{question_id}_*.json"))
    return [
        path for path in raw
        if not path.name.endswith(".payload.json")
    ]


def _latest_payload_summary(payload_dir: Path, question_id: str) -> dict[str, Any] | None:
    candidates = _candidate_payload_paths(payload_dir, question_id)
    if not candidates:
        return None
    path = candidates[-1]
    data = _load_json(path)
    summary = data.get("summary", {})
    raw_payload_path = None
    if path.name.endswith(".meta.json"):
        payload_path = path.with_name(path.name.replace(".meta.json", ".payload.json"))
        if payload_path.exists():
            raw_payload_path = str(payload_path)
    return {
        "path": str(path),
        "raw_payload_path": raw_payload_path,
        "engine_log_path": str(path.with_name(path.name.rsplit(".", 2)[0] + ".engine.log"))
        if path.name.endswith(".meta.json")
        else str(path.with_name(path.name.replace(".json", ".engine.log"))),
        "tool_calls_count": summary.get("tool_calls_count"),
        "continuation_count": summary.get("continuation_count"),
        "stop_reason": summary.get("stop_reason"),
        "reader_input_tokens": summary.get("reader_input_tokens"),
        "reader_output_tokens": summary.get("reader_output_tokens"),
        "tokens_injected": summary.get("tokens_injected"),
        "assembled_total_tokens": summary.get("assembled_total_tokens"),
        "tools_used": summary.get("tools_used"),
    }


def _tool_brief(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in tool_calls:
        row = {
            "tool": call.get("tool"),
            "input": call.get("input", {}),
        }
        result_text = call.get("result")
        if isinstance(result_text, str):
            row["result_head"] = result_text[:400]
        rows.append(row)
    return rows


def _build_question_diff(
    old_question: dict[str, Any],
    new_question: dict[str, Any],
    old_payload_dir: Path | None,
    new_payload_dir: Path | None,
) -> dict[str, Any]:
    question_id = old_question["question_id"]
    old_payload = (
        _latest_payload_summary(old_payload_dir, question_id) if old_payload_dir else None
    )
    new_payload = (
        _latest_payload_summary(new_payload_dir, question_id) if new_payload_dir else None
    )
    old_score = _score(old_question)
    new_score = _score(new_question)
    return {
        "question_id": question_id,
        "category": old_question.get("category"),
        "score": {
            "old": old_score,
            "new": new_score,
            "delta": (new_score - old_score) if old_score is not None and new_score is not None else None,
        },
        "question": old_question.get("question"),
        "gold_answer": old_question.get("ideal_response"),
        "hypothesis": {
            "old": old_question.get("hypothesis"),
            "new": new_question.get("hypothesis"),
        },
        "tools": {
            "old": _tool_brief(old_question.get("tool_calls", [])),
            "new": _tool_brief(new_question.get("tool_calls", [])),
        },
        "payload_summary": {
            "old": old_payload,
            "new": new_payload,
        },
    }


def _build_compare_report(
    old_run: dict[str, Any],
    new_run: dict[str, Any],
    old_results_path: Path,
    new_results_path: Path,
    old_payload_dir: Path | None,
    new_payload_dir: Path | None,
) -> dict[str, Any]:
    old_questions = _question_map(old_run)
    new_questions = _question_map(new_run)
    question_ids = sorted(set(old_questions) & set(new_questions))

    per_category: dict[str, dict[str, float | None]] = {}
    for category, info in old_run.get("summary", {}).get("per_category", {}).items():
        old_avg = info.get("avg_score")
        new_avg = new_run.get("summary", {}).get("per_category", {}).get(category, {}).get("avg_score")
        per_category[category] = {
            "old": old_avg,
            "new": new_avg,
            "delta": (new_avg - old_avg) if old_avg is not None and new_avg is not None else None,
        }

    questions = [
        _build_question_diff(
            old_questions[question_id],
            new_questions[question_id],
            old_payload_dir,
            new_payload_dir,
        )
        for question_id in question_ids
    ]

    regressions = sorted(
        [
            question for question in questions
            if (question["score"]["delta"] or 0.0) < 0
        ],
        key=lambda question: question["score"]["delta"],
    )
    improvements = sorted(
        [
            question for question in questions
            if (question["score"]["delta"] or 0.0) > 0
        ],
        key=lambda question: question["score"]["delta"],
        reverse=True,
    )

    return {
        "old_run": {
            "path": str(old_results_path),
            "payload_dir": str(old_payload_dir) if old_payload_dir else None,
            "overall_score": old_run.get("summary", {}).get("overall_score"),
            "date": old_run.get("date"),
            "reader_model": old_run.get("config", {}).get("reader_model"),
        },
        "new_run": {
            "path": str(new_results_path),
            "payload_dir": str(new_payload_dir) if new_payload_dir else None,
            "overall_score": new_run.get("summary", {}).get("overall_score"),
            "date": new_run.get("date"),
            "reader_model": new_run.get("config", {}).get("reader_model"),
        },
        "overall_score_delta": (
            new_run.get("summary", {}).get("overall_score", 0.0)
            - old_run.get("summary", {}).get("overall_score", 0.0)
        ),
        "per_category": per_category,
        "questions": questions,
        "largest_regressions": [
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "delta": question["score"]["delta"],
            }
            for question in regressions[:10]
        ],
        "largest_improvements": [
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "delta": question["score"]["delta"],
            }
            for question in improvements[:10]
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# BEAM Run Comparison")
    lines.append("")
    lines.append(
        f"- Old overall: {report['old_run']['overall_score']:.3f} "
        f"({report['old_run']['path']})"
    )
    lines.append(
        f"- New overall: {report['new_run']['overall_score']:.3f} "
        f"({report['new_run']['path']})"
    )
    lines.append(f"- Delta: {report['overall_score_delta']:+.3f}")
    lines.append("")
    lines.append("## Category Deltas")
    lines.append("")
    for category, info in sorted(report["per_category"].items()):
        lines.append(
            f"- `{category}`: {info['old']:.3f} -> {info['new']:.3f} ({info['delta']:+.3f})"
        )
    lines.append("")
    lines.append("## Largest Regressions")
    lines.append("")
    for row in report["largest_regressions"]:
        lines.append(
            f"- `{row['question_id']}` [{row['category']}]: {row['delta']:+.3f}"
        )
    lines.append("")
    lines.append("## Largest Improvements")
    lines.append("")
    for row in report["largest_improvements"]:
        lines.append(
            f"- `{row['question_id']}` [{row['category']}]: {row['delta']:+.3f}"
        )
    lines.append("")
    lines.append("## Per-Question Tool Diffs")
    lines.append("")
    for question in report["questions"]:
        lines.append(
            f"### {question['question_id']} ({question['category']}) "
            f"{question['score']['old']:.3f} -> {question['score']['new']:.3f} "
            f"({question['score']['delta']:+.3f})"
        )
        old_summary = question["payload_summary"]["old"] or {}
        new_summary = question["payload_summary"]["new"] or {}
        lines.append(
            f"- Stop reason: `{old_summary.get('stop_reason')}` -> `{new_summary.get('stop_reason')}`"
        )
        lines.append(
            f"- Reader input tokens: `{old_summary.get('reader_input_tokens')}` -> "
            f"`{new_summary.get('reader_input_tokens')}`"
        )
        lines.append(
            f"- Tool calls: `{old_summary.get('tool_calls_count')}` -> "
            f"`{new_summary.get('tool_calls_count')}`"
        )
        lines.append("- Old tools:")
        for tool in question["tools"]["old"]:
            lines.append(f"  - `{tool['tool']}` {json.dumps(tool['input'], sort_keys=True)}")
        lines.append("- New tools:")
        for tool in question["tools"]["new"]:
            lines.append(f"  - `{tool['tool']}` {json.dumps(tool['input'], sort_keys=True)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two BEAM runs and summarize per-question behavior changes."
    )
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--old-payload-dir", type=Path, default=None)
    parser.add_argument("--new-payload-dir", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    parsed = parser.parse_args()

    old_run = _load_json(parsed.old_results)
    new_run = _load_json(parsed.new_results)
    report = _build_compare_report(
        old_run,
        new_run,
        parsed.old_results,
        parsed.new_results,
        parsed.old_payload_dir,
        parsed.new_payload_dir,
    )
    markdown = _render_markdown(report)

    if parsed.json_output:
        parsed.json_output.write_text(json.dumps(report, indent=2))
    if parsed.md_output:
        parsed.md_output.write_text(markdown)
    if not parsed.json_output and not parsed.md_output:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
