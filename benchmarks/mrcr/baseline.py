"""Full-context baseline: send entire MRCR conversation to a model."""

from __future__ import annotations

import json as _json
import logging
import time
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.longmemeval.auth import resolve_provider_token
from benchmarks.longmemeval.cost import BudgetTracker
from benchmarks.longmemeval.openai_codex import (
    codex_simple_text_completion,
    extract_output_text,
    extract_usage,
)

from .dataset import MRCRQuestion

logger = logging.getLogger(__name__)


def _build_messages(q: MRCRQuestion) -> list[dict]:
    """Build the full message list for the baseline API call.

    Returns the original conversation messages plus the final question message,
    all as-is from the MRCR dataset.
    """
    messages = list(q.messages)
    messages.append({"role": "user", "content": q.question_message})
    return messages


def _dump_baseline_payload(
    *,
    question_id: str,
    question_message: str,
    gold_answer: str,
    hypothesis: str,
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
    elapsed_s: float,
    cache_dir: Path | None = None,
) -> None:
    """Save the full baseline payload for autopsy analysis."""
    if not cache_dir:
        cache_dir = Path(__file__).parent / "cache"
    out_dir = cache_dir / question_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"baseline_payload_log_{ts}.json"

    payload = {
        "question_id": question_id,
        "question": question_message,
        "gold_answer": gold_answer,
        "hypothesis": hypothesis,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
    }
    try:
        out_path.write_text(_json.dumps(payload, indent=2, default=str))
        logger.info("Baseline [%s]: payload log saved to %s", question_id, out_path)
    except Exception as e:
        logger.warning("Baseline [%s]: failed to save payload log: %s", question_id, e)


def run_baseline(
    question: MRCRQuestion,
    budget: BudgetTracker,
    model: str = "claude-sonnet-4-5-20250929",
    provider: str = "anthropic",
    api_key: str | None = None,
    auth_mode: str = "auto",
    cache_dir: Path | None = None,
) -> dict:
    """Run full-context baseline for a single MRCR question.

    Sends the entire conversation + question to the model as a multi-turn
    messages payload.  The model must prepend ``random_string_to_prepend``
    before its answer.

    Returns dict with: hypothesis, input_tokens, output_tokens, cost, elapsed_s.
    """
    token = resolve_provider_token(provider, explicit_token=api_key, auth_mode=auth_mode)
    if not token:
        raise ValueError(f"No auth token found for provider {provider}")

    messages = _build_messages(question)
    est_chars = sum(len(m.get("content", "")) for m in messages)
    logger.info(
        "Baseline [%s]: sending %d messages (~%d chars, ~%d tokens) to %s/%s",
        question.question_id, len(messages), est_chars, est_chars // 4, provider, model,
    )

    t0 = time.time()

    if provider == "anthropic":
        # Anthropic requires alternating user/assistant. MRCR data is already
        # in that format. We pass the conversation as messages with a system
        # instruction about the prepend requirement.
        system_text = (
            "You are a helpful assistant. When answering, you MUST prepend "
            f"the following string exactly before your answer: {question.random_string}"
        )
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
            "max_tokens": 8192,
            "temperature": 0.0,
            "system": system_text,
            "messages": messages,
        }
        url = "https://api.anthropic.com/v1/messages"

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

    elif provider == "openai":
        _is_reasoning = model.startswith(("gpt-5", "o3", "o4"))
        # Inject prepend instruction into the system message
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. When answering, you MUST prepend "
                f"the following string exactly before your answer: {question.random_string}"
            ),
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [system_msg] + messages,
        }
        if not _is_reasoning:
            payload["temperature"] = 0.0
        url = "https://api.openai.com/v1/chat/completions"

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
            raise RuntimeError(f"Baseline API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choices = data.get("choices", [])
        hypothesis = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

    elif provider in {"openai-codex", "openai_codex"}:
        # Flatten messages into a single prompt for Codex
        prompt_parts = []
        for m in messages:
            prompt_parts.append(f"{m['role'].capitalize()}: {m.get('content', '')}")
        prompt = "\n\n".join(prompt_parts)
        instructions = (
            "You are a helpful assistant. When answering, you MUST prepend "
            f"the following string exactly before your answer: {question.random_string}"
        )
        data = codex_simple_text_completion(
            token=token,
            model=model,
            prompt=prompt,
            instructions=instructions,
            max_output_tokens=8192,
            temperature=0.0,
            timeout_s=300.0,
        )
        hypothesis = extract_output_text(data)
        input_tokens, output_tokens = extract_usage(data)

    elif provider == "gemini":
        headers = {"Content-Type": "application/json"}
        # Convert to Gemini format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload = {
            "systemInstruction": {
                "parts": [{"text": (
                    "You are a helpful assistant. When answering, you MUST prepend "
                    f"the following string exactly before your answer: {question.random_string}"
                )}],
            },
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.0,
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={token}"

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
            raise RuntimeError(f"Baseline API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        hypothesis = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

    else:
        raise ValueError(f"Unsupported baseline provider: {provider}")

    elapsed = time.time() - t0

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

    result = {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": entry.cost_usd,
        "elapsed_s": round(elapsed, 1),
    }

    _dump_baseline_payload(
        question_id=question.question_id,
        question_message=question.question_message,
        gold_answer=question.answer,
        hypothesis=hypothesis,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider=provider,
        model=model,
        elapsed_s=result["elapsed_s"],
        cache_dir=cache_dir,
    )

    return result
