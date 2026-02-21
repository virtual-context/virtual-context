"""Full-context baseline: send entire haystack to Sonnet."""

from __future__ import annotations

import logging
import os
import time

import httpx

from .cost import BudgetTracker
from .dataset import LongMemEvalQuestion

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def _format_haystack(q: LongMemEvalQuestion) -> str:
    """Format haystack using LongMemEval's exact prompt template."""
    parts = [
        "I will give you several history chats between you and a user.",
        "Please answer the question based on the relevant chat history.",
        "",
        "History Chats:",
    ]

    for i, (session, date) in enumerate(zip(q.haystack_sessions, q.haystack_dates), 1):
        parts.append("")
        parts.append(f"### Session {i}:")
        parts.append(f"Session Date: {date}")
        parts.append("Session Content:")
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")

    parts.append("")
    parts.append(f"Current Date: {q.question_date}")
    parts.append(f"Question: {q.question}")
    parts.append("Answer:")

    return "\n".join(parts)


def run_baseline(
    question: LongMemEvalQuestion,
    budget: BudgetTracker,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
) -> dict:
    """Run full-context baseline for a single question.

    Returns dict with: hypothesis, input_tokens, output_tokens, cost, elapsed_s.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    formatted = _format_haystack(question)
    logger.info(
        "Baseline [%s]: sending ~%d chars (~%d tokens) to %s",
        question.question_id, len(formatted), len(formatted) // 4, model,
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": formatted}],
    }

    t0 = time.time()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(API_URL, headers=headers, json=payload)

    elapsed = time.time() - t0

    if resp.status_code != 200:
        logger.error("Baseline API error %d: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Baseline API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    text_parts = [
        block["text"] for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    hypothesis = "\n".join(text_parts)

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    entry = budget.record(
        label=f"baseline:{question.question_id}",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    logger.info(
        "Baseline [%s]: %d in / %d out tokens, $%.4f, %.1fs",
        question.question_id, input_tokens, output_tokens, entry.cost_usd, elapsed,
    )

    return {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": entry.cost_usd,
        "elapsed_s": round(elapsed, 1),
    }
