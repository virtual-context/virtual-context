"""Running tally for LongMemEval benchmark question attempts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TALLY_FILENAME = "running_tally.json"


def _is_results_payload(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("report_type") == "autopsy_report":
        return False
    if not isinstance(data.get("questions"), list):
        return False
    questions = data["questions"]
    if not questions:
        return False
    first = questions[0]
    if not isinstance(first, dict):
        return False
    return "question_id" in first and isinstance(data.get("config"), dict)


def _verdict(result_obj: Any) -> str:
    if not isinstance(result_obj, dict):
        return "missing"
    if "error" in result_obj:
        return "error"
    correct = result_obj.get("correct")
    if correct is True:
        return "correct"
    if correct is False:
        return "wrong"
    return "unjudged"


def _new_side() -> dict[str, Any]:
    return {
        "attempts": 0,
        "correct": 0,
        "wrong": 0,
        "error": 0,
        "unjudged": 0,
        "latest_verdict": "missing",
    }


def _new_entry(question_type: str | None) -> dict[str, Any]:
    return {
        "question_type": question_type or "",
        "attempts_total": 0,
        "first_seen_file": "",
        "last_seen_file": "",
        "baseline": _new_side(),
        "vc": _new_side(),
        "files": [],
    }


def _apply_side(entry_side: dict[str, Any], verdict: str) -> None:
    if verdict == "missing":
        return
    entry_side["attempts"] += 1
    entry_side["latest_verdict"] = verdict
    if verdict in {"correct", "wrong", "error", "unjudged"}:
        entry_side[verdict] += 1


def refresh_running_tally(base_dir: Path | None = None) -> Path:
    """Rebuild and write running tally across all benchmark result JSON files."""
    root = base_dir or Path(__file__).parent
    tally_path = root / TALLY_FILENAME

    entries: dict[str, dict[str, Any]] = {}
    source_files: list[str] = []
    result_files = sorted(root.glob("*.json"))

    for path in result_files:
        if path.name == TALLY_FILENAME:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not _is_results_payload(data):
            continue

        source_files.append(path.name)
        questions = data.get("questions", [])
        for q in questions:
            if not isinstance(q, dict):
                continue
            qid = str(q.get("question_id", "")).strip()
            if not qid:
                continue

            entry = entries.setdefault(qid, _new_entry(q.get("question_type")))
            entry["attempts_total"] += 1
            entry["question_type"] = entry.get("question_type") or q.get("question_type", "")
            if not entry["first_seen_file"]:
                entry["first_seen_file"] = path.name
            entry["last_seen_file"] = path.name
            if path.name not in entry["files"]:
                entry["files"].append(path.name)

            _apply_side(entry["baseline"], _verdict(q.get("baseline")))
            _apply_side(entry["vc"], _verdict(q.get("vc")))

    baseline_wrong_unique = sum(
        1 for e in entries.values()
        if int(e["baseline"]["wrong"]) > 0
    )
    vc_wrong_unique = sum(
        1 for e in entries.values()
        if int(e["vc"]["wrong"]) > 0
    )

    summary = {
        "source_files_count": len(source_files),
        "question_attempt_records": sum(int(e["attempts_total"]) for e in entries.values()),
        "unique_questions_tried": len(entries),
        "baseline_attempts": sum(int(e["baseline"]["attempts"]) for e in entries.values()),
        "baseline_correct_attempts": sum(int(e["baseline"]["correct"]) for e in entries.values()),
        "baseline_wrong_attempts": sum(int(e["baseline"]["wrong"]) for e in entries.values()),
        "baseline_wrong_unique_questions": baseline_wrong_unique,
        "vc_attempts": sum(int(e["vc"]["attempts"]) for e in entries.values()),
        "vc_correct_attempts": sum(int(e["vc"]["correct"]) for e in entries.values()),
        "vc_wrong_attempts": sum(int(e["vc"]["wrong"]) for e in entries.values()),
        "vc_wrong_unique_questions": vc_wrong_unique,
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": 1,
        "root": str(root),
        "summary": summary,
        "questions": dict(sorted(entries.items(), key=lambda kv: kv[0])),
    }
    tally_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return tally_path

