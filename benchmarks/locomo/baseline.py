"""Full-context baseline for LocOMo: send entire conversation to reader."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.longmemeval.auth import resolve_provider_token
from benchmarks.longmemeval.cost import BudgetTracker
from .dataset import LoCoMoConversation, LoCoMoQuestion
from .vc_runner import _format_question_prompt

logger = logging.getLogger(__name__)


def _format_haystack(conv: LoCoMoConversation) -> str:
    """Format full conversation following LocOMo's prompt template."""
    parts = [
        f"Below is a conversation between two people: {conv.speaker_a} and {conv.speaker_b}.",
        "The conversation takes place over multiple days and the date of each "
        "conversation is written at the beginning of the conversation.",
        "",
    ]
    for session in conv.sessions:
        parts.append(f"DATE: {session.date_time}")
        parts.append("CONVERSATION:")
        for turn in session.turns:
            parts.append(f"{turn['speaker']}: {turn.get('text', '')}")
        parts.append("")

    return "\n".join(parts)


def run_baseline(
    conv: LoCoMoConversation,
    question: LoCoMoQuestion,
    budget: BudgetTracker,
    *,
    model: str = "gemini-3-pro-preview",
    provider: str = "gemini",
    auth_mode: str = "auto",
    cache_dir: Path | None = None,
) -> dict:
    """Run full-context baseline for a single LocOMo question."""
    # For standard API providers, prefer the API key env var directly
    # (resolve_provider_token may return an OAuth token that only works with Codex)
    key_env_map = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
                   "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    key_env = key_env_map.get(provider, "")
    token = os.environ.get(key_env, "") if key_env else ""
    if not token:
        token = resolve_provider_token(provider, explicit_token=None, auth_mode=auth_mode)
    if not token:
        raise ValueError(f"{key_env or provider} auth not found")

    haystack = _format_haystack(conv)
    q_prompt = _format_question_prompt(question, conv)
    formatted = haystack + "\n" + q_prompt

    logger.info(
        "Baseline [%s]: ~%d tokens to %s/%s",
        question.question_id, len(formatted) // 4, provider, model,
    )

    if provider == "anthropic":
        # OAuth tokens (sk-ant-oat*) use Bearer auth + beta header
        if token.startswith("sk-ant-oat"):
            headers = {
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "content-type": "application/json",
            }
        else:
            headers = {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": formatted}],
        }
        url = "https://api.anthropic.com/v1/messages"
    elif provider in ("openai", "openrouter"):
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_completion_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": formatted}],
        }
        url = "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter" else "https://api.openai.com/v1/chat/completions"
    elif provider == "gemini":
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": formatted}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": 2048},
            },
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={token}"
        )
    else:
        raise ValueError(f"Unsupported baseline provider: {provider}")

    t0 = time.time()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        for _retry in range(5):
            if resp.status_code not in (429, 503):
                break
            wait = min(10 * (2 ** _retry), 120)
            logger.warning("Baseline retry %d (HTTP %d), waiting %ds", _retry + 1, resp.status_code, wait)
            time.sleep(wait)
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            logger.error("Baseline API error %d: %s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Baseline API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
    elapsed = time.time() - t0

    if provider == "anthropic":
        text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        hypothesis = "\n".join(text_parts)
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif provider == "openai":
        choices = data.get("choices", [])
        hypothesis = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    else:  # gemini
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        hypothesis = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

    entry = budget.record(
        label=f"baseline:{question.question_id}",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    logger.info(
        "Baseline [%s]: %d in / %d out, $%.4f, %.1fs",
        question.question_id, input_tokens, output_tokens, entry.cost_usd, elapsed,
    )

    # Dump baseline payload
    if cache_dir:
        payloads_dir = (cache_dir / conv.conv_id / "baseline_payloads")
        payloads_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = payloads_dir / f"{question.question_id}_{ts}.json"
        try:
            out_path.write_text(json.dumps({
                "question_id": question.question_id,
                "question": question.question,
                "gold_answer": question.answer,
                "category": question.category,
                "hypothesis": hypothesis,
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_s": round(elapsed, 1),
            }, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save baseline payload: %s", e)

    return {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": entry.cost_usd,
        "elapsed_s": round(elapsed, 1),
    }
