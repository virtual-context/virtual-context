"""Autopsy report generation for LongMemEval benchmark outputs."""

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


def _is_tool_result_useful(tool_name: str, raw_result: Any) -> tuple[bool | None, str]:
    """Heuristic usefulness classification for tool-call outputs."""
    if not isinstance(raw_result, str):
        return None, "Unknown usefulness: non-string result payload."

    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return None, "Unknown usefulness: result is not JSON."

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

    return None, "Unknown usefulness: no heuristic for this tool."


def _summarize_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    tool_name = str(call.get("tool", ""))
    raw_result = call.get("result")
    useful, reason = _is_tool_result_useful(tool_name, raw_result)

    parsed_result = None
    if isinstance(raw_result, str):
        try:
            parsed_result = json.loads(raw_result)
        except json.JSONDecodeError:
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


def _load_reader_payload(question_id: str, cache_dir: Path | None) -> dict[str, Any] | None:
    """Load the payload_log.json for a question and extract the reader payload.

    Returns a structured dict with:
    - system_prompt: the full system instruction text sent to the reader
    - user_message: the user message (question) sent to the reader
    - rounds: list of request/response round-trips with extracted text
    """
    if not cache_dir or not question_id:
        return None

    q_dir = cache_dir / question_id
    # Find newest timestamped payload log, fall back to legacy name
    candidates = sorted(q_dir.glob("payload_log_*.json"))
    if candidates:
        payload_path = candidates[-1]
    else:
        payload_path = q_dir / "payload_log.json"
    if not payload_path.exists():
        return None

    try:
        payload = json.loads(payload_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    http_conv = payload.get("http_conversation", [])
    if not http_conv:
        return None

    # Extract system prompt and user message from the first request
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
            # Extract system instruction (varies by provider format)
            si = body.get("system_instruction") or body.get("system") or ""
            if isinstance(si, dict):
                # Gemini format: {"parts": [{"text": "..."}]}
                parts = si.get("parts", [])
                si_text = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in parts
                )
            elif isinstance(si, list):
                # Anthropic format: [{"type": "text", "text": "..."}]
                si_text = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in si
                )
            else:
                si_text = str(si) if si else ""

            if step == 1 and si_text:
                system_prompt = si_text

            # Extract user message from contents/messages
            contents = body.get("contents") or body.get("messages") or []
            user_texts = []
            for msg in (contents if isinstance(contents, list) else []):
                role = msg.get("role", "")
                if role == "user":
                    # Gemini: parts[].text, Anthropic: content
                    msg_parts = msg.get("parts", [])
                    if msg_parts:
                        for p in msg_parts:
                            if isinstance(p, dict) and p.get("text"):
                                user_texts.append(p["text"])
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        user_texts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("text"):
                                user_texts.append(block["text"])

            if step == 1 and user_texts:
                user_message = "\n".join(user_texts)

            round_entry = {"step": step, "request_user_texts": user_texts}
            # Check if this request includes tool results
            tool_results_in_request = []
            for msg in (contents if isinstance(contents, list) else []):
                msg_parts = msg.get("parts", [])
                for p in (msg_parts if isinstance(msg_parts, list) else []):
                    if isinstance(p, dict) and "functionResponse" in p:
                        fr = p["functionResponse"]
                        tool_results_in_request.append({
                            "tool": fr.get("name", ""),
                            "result": fr.get("response", {}).get("result", ""),
                        })
                # Anthropic format: tool_result blocks
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
            # Find the matching round
            matching = [r for r in rounds if r["step"] == step]
            if not matching:
                rounds.append({"step": step})
                matching = [rounds[-1]]
            round_entry = matching[-1]

            # Extract response text and tool calls
            candidates = body.get("candidates", [])
            for cand in (candidates if isinstance(candidates, list) else []):
                content = cand.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                for p in (parts if isinstance(parts, list) else []):
                    if isinstance(p, dict):
                        if p.get("text"):
                            round_entry["response_text"] = p["text"]
                        if "functionCall" in p:
                            fc = p["functionCall"]
                            round_entry.setdefault("tool_calls_made", []).append({
                                "tool": fc.get("name", ""),
                                "args": fc.get("args", {}),
                            })

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


def _find_latest(directory: Path, prefix: str) -> Path | None:
    """Find the newest timestamped file matching prefix_*.json, or fall back to prefix.json."""
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
    vc_total_injected = 0.0

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
        vc_assembled_total = _safe_num(vc.get("assembled_total_tokens"))

        baseline_total_cost += baseline_cost
        vc_total_cost += vc_cost
        baseline_total_input += baseline_input
        baseline_total_output += baseline_output
        vc_total_input += vc_input
        vc_total_output += vc_output
        vc_total_injected += vc_injected

        tool_calls = vc.get("tool_calls", []) if isinstance(vc.get("tool_calls"), list) else []
        tool_loop = [_summarize_tool_call(c, i + 1) for i, c in enumerate(tool_calls) if isinstance(c, dict)]

        # Chain analytics
        from .chain_analysis import analyze_tool_chain
        chain_analysis = analyze_tool_chain(tool_calls)

        # Load the full reader payload from the cache
        qid = q.get("question_id", "")
        reader_payload = _load_reader_payload(qid, cache_dir)

        # Resolve payload log paths — find latest timestamped files
        base_cache = cache_dir or Path("benchmarks/longmemeval/cache")
        q_dir = base_cache / qid
        vc_log = _find_latest(q_dir, "payload_log") if q_dir.exists() else None
        bl_log = _find_latest(q_dir, "baseline_payload_log") if q_dir.exists() else None
        vc_payload_log_path = str(vc_log) if vc_log else str(q_dir / "payload_log.json")
        baseline_payload_log_path = str(bl_log) if bl_log else str(q_dir / "baseline_payload_log.json")

        question_reports.append(
            {
                "question_id": qid,
                "question_type": q.get("question_type"),
                "question": q.get("question"),
                "gold_answer": q.get("answer"),
                "reader_payload": reader_payload,
                "payload_paths": {
                    "vc": vc_payload_log_path,
                    "baseline": baseline_payload_log_path,
                },
                "baseline": {
                    "answer": baseline.get("hypothesis"),
                    "correct": baseline.get("correct"),
                    "judge_explanation": baseline.get("judge_explanation"),
                    "judge_yes_votes": baseline.get("judge_yes_votes"),
                    "judge_total_votes": baseline.get("judge_total_votes"),
                    "judge_vote_mode": baseline.get("judge_vote_mode"),
                    "token_usage": {
                        "input_tokens": baseline_input,
                        "output_tokens": baseline_output,
                        "total_tokens": baseline_input + baseline_output,
                    },
                    "cost_usd": baseline_cost,
                    "elapsed_s": _safe_num(baseline.get("elapsed_s")),
                    "error": baseline.get("error"),
                },
                "vc": {
                    "answer": vc.get("hypothesis"),
                    "correct": vc.get("correct"),
                    "judge_explanation": vc.get("judge_explanation"),
                    "judge_yes_votes": vc.get("judge_yes_votes"),
                    "judge_total_votes": vc.get("judge_total_votes"),
                    "judge_vote_mode": vc.get("judge_vote_mode"),
                    "token_usage": {
                        "reader_input_tokens": vc_input,
                        "reader_output_tokens": vc_output,
                        "reader_total_tokens": vc_input + vc_output,
                        "tokens_injected": vc_injected,
                        "assembled_total_tokens": vc_assembled_total,
                        "internal_input_tokens": _safe_num(vc.get("haiku_input_tokens")),
                        "internal_output_tokens": _safe_num(vc.get("haiku_output_tokens")),
                        "internal_calls": _safe_num(vc.get("haiku_calls")),
                    },
                    "cost_usd": vc_cost,
                    "elapsed_s": _safe_num(vc.get("elapsed_s")),
                    "compaction_events": vc.get("compaction_events"),
                    "tags_matched": vc.get("tags_matched", []),
                    "tool_loop": {
                        "continuations": vc.get("continuation_count"),
                        "stop_reason": vc.get("stop_reason"),
                        "calls": tool_loop,
                        "chain_analysis": chain_analysis,
                    },
                    "error": vc.get("error"),
                },
                "haystack_tokens_est": q.get("haystack_tokens_est"),
                "n_sessions": q.get("n_sessions"),
            }
        )

    return {
        "report_type": "autopsy_report",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": results_data.get("benchmark", "longmemeval"),
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
            "vc_total_tokens_injected": vc_total_injected,
        },
        "questions": question_reports,
    }


def render_autopsy_markdown(autopsy: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# LongMemEval Autopsy Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {autopsy.get('generated_at_utc', '')}")
    lines.append(f"- Benchmark: {autopsy.get('benchmark', '')}")
    lines.append(f"- Run date: {autopsy.get('run_date', '')}")
    lines.append(f"- Total run cost: ${_safe_num(autopsy.get('total_run_cost_usd')):.4f}")
    cc = autopsy.get("cost_comparison", {})
    lines.append(f"- Baseline total cost: ${_safe_num(cc.get('baseline_total_cost_usd')):.4f}")
    lines.append(f"- VC total cost: ${_safe_num(cc.get('vc_total_cost_usd')):.4f}")
    lines.append(f"- Cost delta (VC - Baseline): ${_safe_num(cc.get('delta_vc_minus_baseline_usd')):.4f}")
    lines.append("")

    for q in autopsy.get("questions", []):
        lines.append(f"## Question {q.get('question_id', '')} ({q.get('question_type', '')})")
        lines.append("")
        lines.append(f"- Original question: {q.get('question', '')}")
        lines.append(f"- Gold answer: {q.get('gold_answer', '')}")
        lines.append(f"- Haystack size: ~{q.get('haystack_tokens_est', 0)} tokens, {q.get('n_sessions', 0)} sessions")
        pp = q.get("payload_paths", {})
        if pp:
            lines.append(f"- VC payload log: `{pp.get('vc', '')}`")
            lines.append(f"- Baseline payload log: `{pp.get('baseline', '')}`")
        lines.append("")

        baseline = q.get("baseline", {})
        bt = baseline.get("token_usage", {})
        lines.append("### Baseline")
        lines.append(f"- Full Answer: {baseline.get('answer', '')}")
        lines.append(f"- Correct: {baseline.get('correct', None)}")
        lines.append(f"- Judge explanation: {baseline.get('judge_explanation', '')}")
        baseline_total_votes = int(_safe_num(baseline.get("judge_total_votes")))
        if baseline_total_votes > 0:
            lines.append(
                f"- Judge votes: {int(_safe_num(baseline.get('judge_yes_votes')))}"
                f"/{baseline_total_votes} yes ({baseline.get('judge_vote_mode', '')})"
            )
        lines.append(
            f"- Tokens: in={int(_safe_num(bt.get('input_tokens')))}, "
            f"out={int(_safe_num(bt.get('output_tokens')))}, "
            f"total={int(_safe_num(bt.get('total_tokens')))}"
        )
        lines.append(f"- Cost: ${_safe_num(baseline.get('cost_usd')):.4f}")
        lines.append(f"- Elapsed: {_safe_num(baseline.get('elapsed_s')):.1f}s")
        if baseline.get("error"):
            lines.append(f"- Error: {baseline.get('error')}")
        lines.append("")

        # Reader payload summary
        rp = q.get("reader_payload")
        if rp:
            lines.append(f"- User question sent to reader: `{rp.get('user_message', '(not available)')}`")
        lines.append("")

        vc = q.get("vc", {})
        vt = vc.get("token_usage", {})
        lines.append("### VC")
        lines.append(f"- Full Answer: {vc.get('answer', '')}")
        lines.append(f"- Correct: {vc.get('correct', None)}")
        lines.append(f"- Judge explanation: {vc.get('judge_explanation', '')}")
        vc_total_votes = int(_safe_num(vc.get("judge_total_votes")))
        if vc_total_votes > 0:
            lines.append(
                f"- Judge votes: {int(_safe_num(vc.get('judge_yes_votes')))}"
                f"/{vc_total_votes} yes ({vc.get('judge_vote_mode', '')})"
            )
        lines.append(
            f"- Reader tokens: in={int(_safe_num(vt.get('reader_input_tokens')))}, "
            f"out={int(_safe_num(vt.get('reader_output_tokens')))}, "
            f"total={int(_safe_num(vt.get('reader_total_tokens')))}"
        )
        lines.append(
            f"- VC internal tokens: in={int(_safe_num(vt.get('internal_input_tokens')))}, "
            f"out={int(_safe_num(vt.get('internal_output_tokens')))}, "
            f"calls={int(_safe_num(vt.get('internal_calls')))}"
        )
        lines.append(f"- Prompt tokens sent (est): {int(_safe_num(vt.get('tokens_injected')))}")
        assembled_total = int(_safe_num(vt.get("assembled_total_tokens")))
        if assembled_total > 0:
            lines.append(f"- Assembled context tokens (debug): {assembled_total}")
        lines.append(f"- Cost: ${_safe_num(vc.get('cost_usd')):.4f}")
        lines.append(f"- Elapsed: {_safe_num(vc.get('elapsed_s')):.1f}s")
        lines.append(f"- Compaction events: {vc.get('compaction_events', 0)}")
        lines.append(f"- Matched tags: {', '.join(vc.get('tags_matched', [])[:10])}")
        if vc.get("error"):
            lines.append(f"- Error: {vc.get('error')}")
        lines.append("")

        tool_loop = vc.get("tool_loop", {})
        lines.append("### Tool Loop Autopsy")
        lines.append(f"- Continuations: {tool_loop.get('continuations', 0)}")
        lines.append(f"- Stop reason: {tool_loop.get('stop_reason', '')}")
        calls = tool_loop.get("calls", [])
        if not calls:
            lines.append("- Calls: none")
        for c in calls:
            lines.append(f"- Call #{c.get('index')}: `{c.get('tool', '')}`")
            lines.append(f"- Parameters: `{json.dumps(c.get('parameters', {}), ensure_ascii=True)}`")
            lines.append(f"- Duration: {_safe_num(c.get('duration_ms')):.1f}ms")
            lines.append(f"- Useful: {c.get('useful', None)}")
            lines.append(f"- Usefulness reason: {c.get('usefulness_reason', '')}")
            lines.append(f"- Returned JSON: `{json.dumps(c.get('returned_json'), ensure_ascii=True)}`")
        lines.append("")

        chain = tool_loop.get("chain_analysis", {})
        if chain and chain.get("total_calls", 0) > 0:
            lines.append("### Chain Analysis")
            lines.append(f"- Pattern: {chain.get('chain_pattern', 'unknown')}")
            utypes = chain.get("unique_tool_types", [])
            lines.append(f"- Tool types: {', '.join(utypes)} ({chain.get('unique_tool_type_count', 0)} unique)")
            lines.append(f"- Useful/wasted: {chain.get('useful_calls', 0)}/{chain.get('wasted_calls', 0)}")
            lines.append(f"- Tokens freed: {chain.get('tokens_freed_total', 0):,} | Tokens added: {chain.get('tokens_added_total', 0):,}")
            if chain.get("has_strategy_pivot"):
                lines.append("- Strategy pivot: Yes")
            if chain.get("has_collapse_then_expand"):
                lines.append("- Collapse-then-expand: Yes")
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
