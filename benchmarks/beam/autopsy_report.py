"""Autopsy report generation for BEAM benchmark outputs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_tool_result_useful(tool_name: str, raw_result: Any) -> tuple[bool | None, str]:
    """Heuristic usefulness classification for tool-call outputs."""
    if not isinstance(raw_result, str):
        return None, "Unknown usefulness: non-string result payload."

    try:
        payload = json.loads(raw_result)
    except (json.JSONDecodeError, ValueError):
        # Not JSON — check for plain-text indicators
        if raw_result.strip():
            return True, "Useful: non-empty plain-text result."
        return False, "Not useful: empty result."

    if tool_name == "vc_find_quote":
        found = payload.get("found")
        results = payload.get("results")
        if found is True and isinstance(results, list) and len(results) > 0:
            return True, "Useful: quote search returned one or more matches."
        if found is False or (isinstance(results, list) and len(results) == 0):
            return False, "Not useful: quote search returned no matches."
        return None, "Unknown usefulness: quote-search shape was unexpected."

    if tool_name in {"vc_expand_topic", "vc_recall_all", "vc_remember_when"}:
        if payload:
            return True, "Useful: retrieval tool returned non-empty payload."
        return False, "Not useful: retrieval tool returned empty payload."

    if tool_name == "vc_query_facts":
        facts = payload.get("facts", payload.get("results", []))
        if isinstance(facts, list) and len(facts) > 0:
            return True, f"Useful: query_facts returned {len(facts)} fact(s)."
        return False, "Not useful: query_facts returned no matches."

    return None, "Unknown usefulness: no heuristic for this tool."


def _summarize_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    tool_name = str(call.get("tool", ""))
    raw_result = call.get("result", "")
    useful, reason = _is_tool_result_useful(tool_name, raw_result)

    parsed_result = None
    if isinstance(raw_result, str):
        try:
            parsed_result = json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            parsed_result = None

    return {
        "index": index,
        "tool": tool_name,
        "parameters": call.get("input", {}),
        "duration_ms": _safe_num(call.get("duration_ms")),
        "returned_raw": raw_result,
        "returned_json": parsed_result,
        "useful": useful,
        "usefulness_reason": reason,
    }


def _find_latest_payload(payloads_dir: Path, question_id: str) -> Path | None:
    """Find the newest timestamped payload log for a question."""
    # Prefer .payload.json (new format); fall back to .json (legacy)
    candidates = sorted(payloads_dir.glob(f"{question_id}_*.payload.json"))
    if not candidates:
        candidates = sorted(payloads_dir.glob(f"{question_id}_*.json"))
    return candidates[-1] if candidates else None


def _load_reader_payload(question_id: str, conv_cache_dir: Path) -> dict[str, Any] | None:
    """Load the payload log for a question and extract the reader payload.

    Returns structured dict with system_prompt, user_message, rounds.
    """
    payloads_dir = conv_cache_dir / "payloads"
    if not payloads_dir.exists():
        return None

    payload_path = _find_latest_payload(payloads_dir, question_id)
    if not payload_path or not payload_path.exists():
        return None

    try:
        payload = json.loads(payload_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # New format: payload file is a bare array of request/response entries.
    # Legacy format: dict with "http_conversation" key.
    if isinstance(payload, list):
        http_conv = payload
    else:
        http_conv = payload.get("http_conversation", [])
    if not http_conv:
        return None

    system_prompt = ""
    user_message = ""
    rounds: list[dict[str, Any]] = []

    for entry in http_conv:
        direction = entry.get("direction", "")
        step = entry.get("step", 0)
        body = entry.get("body", {})
        if not isinstance(body, dict):
            continue

        if direction == "REQUEST":
            # Extract system instruction (Anthropic format)
            si = body.get("system") or ""
            if isinstance(si, list):
                si_text = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in si
                )
            elif isinstance(si, str):
                si_text = si
            else:
                si_text = str(si) if si else ""

            if step == 1 and si_text:
                system_prompt = si_text

            # Extract user message
            contents = body.get("messages") or []
            user_texts = []
            for msg in (contents if isinstance(contents, list) else []):
                role = msg.get("role", "")
                if role == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        user_texts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("text"):
                                user_texts.append(block["text"])

            if step == 1 and user_texts:
                user_message = "\n".join(user_texts)

            round_entry: dict[str, Any] = {"step": step, "request_user_texts": user_texts}

            # Check for tool results in request
            tool_results_in_request = []
            for msg in (contents if isinstance(contents, list) else []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_results_in_request.append({
                                "tool": block.get("name", ""),
                                "result": block.get("content", ""),
                            })
            if tool_results_in_request:
                round_entry["tool_results_sent"] = tool_results_in_request

            rounds.append(round_entry)

        elif direction == "RESPONSE":
            matching = [r for r in rounds if r["step"] == step]
            if not matching:
                rounds.append({"step": step})
                matching = [rounds[-1]]
            round_entry = matching[-1]

            # Anthropic format
            resp_content = body.get("content", [])
            if isinstance(resp_content, list):
                for block in resp_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            round_entry["response_text"] = block.get("text", "")
                        if block.get("type") == "tool_use":
                            round_entry.setdefault("tool_calls_made", []).append({
                                "tool": block.get("name", ""),
                                "args": block.get("input", {}),
                            })

    return {
        "system_prompt": system_prompt,
        "user_message": user_message,
        "system_prompt_chars": len(system_prompt),
        "rounds": rounds,
    }


def _load_store_diagnostics(conv_cache_dir: Path, question: dict[str, Any]) -> dict[str, Any]:
    """Load diagnostic data from the VC store for a question.

    Checks for relevant facts and segments related to the question's gold answer.
    """
    db_path = conv_cache_dir / "store.db"
    if not db_path.exists():
        return {"available": False}

    diag: dict[str, Any] = {"available": True}

    try:
        conn = sqlite3.connect(str(db_path))

        # Count total facts and segments
        total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        total_segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        total_tags = conn.execute("SELECT COUNT(DISTINCT tag) FROM tag_summaries").fetchone()[0]
        superseded = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]

        diag["store_stats"] = {
            "total_facts": total_facts,
            "total_segments": total_segments,
            "total_tags": total_tags,
            "superseded_facts": superseded,
        }

        # Search for gold answer keywords in facts
        gold = question.get("ideal_response", "") or ""
        hypothesis = question.get("hypothesis", "") or ""

        # Extract key terms (numbers, proper nouns) from gold answer
        import re
        numbers_in_gold = re.findall(r'\d+', gold)
        numbers_in_hypothesis = re.findall(r'\d+', hypothesis)

        fact_hits = []
        rows = conn.execute(
            "SELECT id, subject, verb, object, status, superseded_by FROM facts"
        ).fetchall()
        for fid, subj, verb, obj, status, superseded_by in rows:
            fact_text = f"{subj} {verb} {obj}".lower()
            gold_lower = gold.lower()
            # Check if any gold answer keywords appear in this fact
            matched = False
            for num in numbers_in_gold:
                if num in fact_text:
                    matched = True
                    break
            if not matched:
                # Check significant words from gold (>4 chars)
                for word in gold_lower.split():
                    if len(word) > 4 and word in fact_text:
                        matched = True
                        break
            if matched:
                fact_hits.append({
                    "id": fid[:12] + "...",
                    "triple": f"{subj} | {verb} | {obj}",
                    "status": status,
                    "superseded_by": superseded_by,
                })

        diag["relevant_facts"] = fact_hits
        diag["relevant_facts_count"] = len(fact_hits)

        # Search segments for gold answer keywords
        segment_hits = []
        seg_rows = conn.execute(
            "SELECT ref, primary_tag, summary FROM segments"
        ).fetchall()
        for ref, tag, summary in seg_rows:
            summary_lower = summary.lower()
            gold_lower = gold.lower()
            matched_keywords = []
            for num in numbers_in_gold + numbers_in_hypothesis:
                if num in summary_lower:
                    matched_keywords.append(num)
            if matched_keywords:
                segment_hits.append({
                    "tag": tag,
                    "matched_keywords": matched_keywords,
                    "summary_preview": summary[:200],
                })

        diag["relevant_segments"] = segment_hits
        diag["relevant_segments_count"] = len(segment_hits)

        conn.close()
    except Exception as e:
        diag["error"] = str(e)

    return diag


def build_autopsy_data(
    results_data: dict[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Build structured autopsy data from BEAM results."""
    conversations = results_data.get("conversations", [])
    chat_size = results_data.get("chat_size", "100K")

    all_scores: list[float] = []
    question_reports: list[dict[str, Any]] = []

    for conv in conversations:
        conv_id = conv.get("conv_id", "")
        ingest_stats = conv.get("ingest_stats", {})

        base_cache = cache_dir or Path("benchmarks/beam/cache")
        conv_cache = base_cache / chat_size / conv_id

        for q in conv.get("questions", []):
            qid = q.get("question_id", "")
            judgment = q.get("judgment", {}) or {}
            score = _safe_num(judgment.get("score"))
            all_scores.append(score)

            tool_calls = q.get("tool_calls", [])
            tool_loop = [
                _summarize_tool_call(c, i + 1)
                for i, c in enumerate(tool_calls)
                if isinstance(c, dict)
            ]

            # Load full reader payload
            reader_payload = _load_reader_payload(qid, conv_cache)

            # Payload log path
            payloads_dir = conv_cache / "payloads"
            payload_path = _find_latest_payload(payloads_dir, qid) if payloads_dir.exists() else None

            # Store diagnostics
            store_diag = _load_store_diagnostics(conv_cache, q)

            question_reports.append({
                "question_id": qid,
                "conv_id": conv_id,
                "category": q.get("category"),
                "difficulty": q.get("difficulty"),
                "question": q.get("question"),
                "gold_answer": q.get("ideal_response"),
                "hypothesis": q.get("hypothesis"),
                "rubric": q.get("rubric", []),
                "payload_path": str(payload_path) if payload_path else "",
                "reader_payload": reader_payload,
                "judgment": {
                    "score": score,
                    "method": judgment.get("method"),
                    "criteria_scores": judgment.get("criteria_scores", []),
                    "criteria_met": judgment.get("criteria_met", []),
                    "criteria_missed": judgment.get("criteria_missed", []),
                },
                "retrieval": {
                    "tags_matched": q.get("tags_matched", []),
                    "tokens_injected": _safe_num(q.get("tokens_injected")),
                    "timings": q.get("timings", {}),
                },
                "tool_loop": {
                    "calls": tool_loop,
                    "call_count": len(tool_loop),
                },
                "store_diagnostics": store_diag,
                "ingest_stats": ingest_stats,
            })

    return {
        "report_type": "beam_autopsy_report",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "beam",
        "chat_size": chat_size,
        "run_date": results_data.get("date"),
        "config": results_data.get("config", {}),
        "summary": results_data.get("summary", {}),
        "elapsed_s": _safe_num(results_data.get("elapsed_s")),
        "budget_spent": _safe_num(results_data.get("budget_spent")),
        "questions": question_reports,
    }


def render_autopsy_markdown(autopsy: dict[str, Any]) -> str:
    """Render autopsy data as human-readable Markdown."""
    lines: list[str] = []
    lines.append("# BEAM Autopsy Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {autopsy.get('generated_at_utc', '')}")
    lines.append(f"- Chat size: {autopsy.get('chat_size', '')}")
    lines.append(f"- Run date: {autopsy.get('run_date', '')}")
    lines.append(f"- Elapsed: {_safe_num(autopsy.get('elapsed_s')):.0f}s")
    lines.append(f"- Budget spent: ${_safe_num(autopsy.get('budget_spent')):.4f}")
    lines.append("")

    # Summary
    summary = autopsy.get("summary", {})
    if summary.get("overall_score") is not None:
        lines.append(f"- Overall score: {summary['overall_score']:.3f}")
    lines.append(f"- Total questions: {summary.get('total_questions', 0)}")
    lines.append(f"- Total judged: {summary.get('total_judged', 0)}")
    per_cat = summary.get("per_category", {})
    if per_cat:
        lines.append("")
        lines.append("| Category | Avg Score | Count |")
        lines.append("|----------|-----------|-------|")
        for cat, info in sorted(per_cat.items()):
            lines.append(f"| {cat} | {info.get('avg_score', 0):.3f} | {info.get('count', 0)} |")
    lines.append("")

    # Config
    config = autopsy.get("config", {})
    if config:
        lines.append("## Config")
        for k, v in config.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # Per-question reports
    for q in autopsy.get("questions", []):
        judgment = q.get("judgment", {})
        score = _safe_num(judgment.get("score"))
        status = "PASS" if score >= 1.0 else ("PARTIAL" if score >= 0.5 else "FAIL")

        lines.append(f"## {q.get('question_id', '')} [{status}] ({q.get('category', '')}, {q.get('difficulty', '')})")
        lines.append("")
        lines.append(f"- **Question**: {q.get('question', '')}")
        lines.append(f"- **Gold answer**: {q.get('gold_answer', '')}")
        lines.append(f"- **VC answer**: {q.get('hypothesis', '')}")
        lines.append(f"- **Score**: {score:.2f} ({judgment.get('method', '')})")
        lines.append("")

        # Rubric criteria
        criteria = judgment.get("criteria_scores", [])
        if criteria:
            lines.append("### Criteria")
            for c in criteria:
                cs = _safe_num(c.get("score"))
                icon = "+" if cs >= 1.0 else ("~" if cs >= 0.5 else "x")
                lines.append(f"- [{icon}] {c.get('criterion', '')} (score={cs:.1f})")
                reason = c.get("reason", "")
                if reason:
                    lines.append(f"  - {reason[:300]}")
            lines.append("")

        # Retrieval
        ret = q.get("retrieval", {})
        lines.append("### Retrieval")
        lines.append(f"- Tokens injected: {int(_safe_num(ret.get('tokens_injected')))}")
        lines.append(f"- Tags matched: {', '.join(ret.get('tags_matched', []))}")
        timings = ret.get("timings", {})
        if timings:
            lines.append(f"- Retrieve time: {_safe_num(timings.get('retrieve_s')):.1f}s")
            lines.append(f"- Query time: {_safe_num(timings.get('query_s')):.1f}s")
        lines.append("")

        # Tool loop
        tl = q.get("tool_loop", {})
        calls = tl.get("calls", [])
        if calls:
            lines.append("### Tool Loop")
            lines.append(f"- Calls: {tl.get('call_count', 0)}")
            lines.append("")
            lines.append("| # | Tool | Parameters | Duration | Useful | Reason |")
            lines.append("|---|------|-----------|----------|--------|--------|")
            for c in calls:
                params = json.dumps(c.get("parameters", {}), ensure_ascii=False)
                if len(params) > 60:
                    params = params[:57] + "..."
                useful_str = str(c.get("useful", "?"))
                reason_str = c.get("usefulness_reason", "")[:60]
                lines.append(
                    f"| {c.get('index')} "
                    f"| `{c.get('tool', '')}` "
                    f"| `{params}` "
                    f"| {_safe_num(c.get('duration_ms')):.0f}ms "
                    f"| {useful_str} "
                    f"| {reason_str} |"
                )
            lines.append("")

            # Show returned data for each call
            for c in calls:
                raw = c.get("returned_raw", "")
                if raw:
                    lines.append(f"**Call #{c.get('index')} result**: `{str(raw)[:300]}`")
            lines.append("")

        # Store diagnostics
        diag = q.get("store_diagnostics", {})
        if diag.get("available"):
            lines.append("### Store Diagnostics")
            stats = diag.get("store_stats", {})
            lines.append(
                f"- Store: {stats.get('total_facts', 0)} facts, "
                f"{stats.get('total_segments', 0)} segments, "
                f"{stats.get('total_tags', 0)} tags, "
                f"{stats.get('superseded_facts', 0)} superseded"
            )

            rel_facts = diag.get("relevant_facts", [])
            if rel_facts:
                lines.append(f"- **Relevant facts found** ({len(rel_facts)}):")
                for f in rel_facts[:5]:
                    sup = f" [SUPERSEDED by {f['superseded_by'][:12]}...]" if f.get("superseded_by") else ""
                    lines.append(f"  - {f['triple']} (status={f['status']}{sup})")
            else:
                lines.append("- **No relevant facts found** (fact extractor missed this data point)")

            rel_segs = diag.get("relevant_segments", [])
            if rel_segs:
                lines.append(f"- **Relevant segments found** ({len(rel_segs)}):")
                for s in rel_segs[:5]:
                    lines.append(f"  - [{s['tag']}] matched: {s['matched_keywords']} — {s['summary_preview'][:120]}...")
            else:
                lines.append("- **No relevant segments found**")
            lines.append("")

        # Reader payload
        rp = q.get("reader_payload")
        if rp:
            lines.append("### Reader Payload")
            lines.append(f"- System prompt: {rp.get('system_prompt_chars', 0):,} chars")
            lines.append(f"- User message: `{rp.get('user_message', '')[:200]}`")
            lines.append(f"- Rounds: {len(rp.get('rounds', []))}")
            lines.append("")

        if q.get("payload_path"):
            lines.append(f"*Full payload log*: `{q['payload_path']}`")
            lines.append("")

        lines.append("---")
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
