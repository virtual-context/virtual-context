"""LocOMo scoring — F1 (deterministic) + LLM judge (semantic)."""

from __future__ import annotations

import logging
import os
import re
import string
import time
from collections import Counter

import httpx
from nltk.stem import PorterStemmer

from .dataset import CATEGORY_NAMES, LoCoMoQuestion

logger = logging.getLogger(__name__)

_ps = PorterStemmer()

# ---------------------------------------------------------------------------
# LLM Judge prompt templates — category-specific
# ---------------------------------------------------------------------------

_JUDGE_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Answer YES if the response is semantically correct — it contains or is equivalent "
    "to the correct answer. Answer NO if it is wrong, missing key information, or "
    "contradicts the correct answer.\n"
    "{extra_rules}"
    "{date_context}"
    "\nQuestion: {question}"
    "\nCorrect Answer: {gold}"
    "\nModel Response: {prediction}"
    "\n\nIs the model response correct? Answer YES or NO only."
)

_JUDGE_RULES: dict[int, str] = {
    1: (  # multi-hop
        "The answer may require combining information from multiple sessions. "
        "If the response contains all required sub-answers (possibly in different order "
        "or wording), it is correct. Accept genuine synonyms (e.g. 'dance competition' "
        "and 'dance festival performance' describe the same type of event). "
        "But do NOT accept substitutions of entirely different events — "
        "'investor event' is NOT 'dance competition'. "
        "Extra sub-answers beyond the gold are NOT wrong, but all gold sub-answers "
        "must be present.\n"
    ),
    2: (  # temporal
        "IMPORTANT: Two date expressions referring to the same calendar date are "
        "equivalent regardless of format. For example, 'May 20, 2023' and "
        "'The saturday before 25 May 2023' and 'last Saturday' (relative to May 25) "
        "all refer to the same date and are ALL correct.\n"
        "Off-by-one tolerance: dates within 1-2 days of the correct answer are CORRECT. "
        "For example, if the gold answer is 'Sunday May 21' and the response says "
        "'May 20' (Saturday), that is CORRECT — same weekend, off by one day. "
        "Similarly, 'July 1' vs 'July 3', or '18 days' vs '19 days' are all correct.\n"
        "Do not penalize differences in date format (ISO vs written vs relative).\n"
    ),
    3: (  # inference / single-hop
        "The response is correct if it captures the same factual information, "
        "even if worded differently.\n"
    ),
    4: (  # open-domain / commonsense
        "The response may combine conversation context with world knowledge. "
        "It is correct if it arrives at the same conclusion as the gold answer.\n"
        "A resolved date (e.g. 'January 27, 2022') is equivalent to a relative date "
        "(e.g. 'a year ago') if they refer to the same time. "
        "Synonyms and paraphrases are acceptable — 'dance festival performance' and "
        "'dance competition' refer to the same type of event. "
        "Extra detail beyond the gold answer is NOT wrong — a more specific correct "
        "answer is still correct.\n"
    ),
    5: (  # adversarial
        "This is an adversarial question — the correct behavior is to identify it as "
        "unanswerable or not mentioned in the conversation. The response is correct if "
        "it indicates the information is not available, not mentioned, or the question "
        "cannot be answered from the conversation.\n"
    ),
}


def _build_judge_prompt(
    question: str, gold: str, prediction: str, category: int,
    conversation_date: str = "",
) -> str:
    """Build the judge prompt with category-specific rules."""
    extra = _JUDGE_RULES.get(category, "")
    date_context = ""
    if conversation_date:
        date_context = (
            f"\nConversation date: {conversation_date}. "
            f"Relative time references like 'a year ago', 'last month' are relative "
            f"to this date. A resolved date (e.g. '2022-01-27') that matches "
            f"a relative reference ('a year ago' from {conversation_date}) is CORRECT.\n"
        )
    return _JUDGE_BASE.format(
        extra_rules=extra, date_context=date_context,
        question=question, gold=gold, prediction=prediction,
    )


def strip_reasoning_preamble(s: str) -> str:
    """Remove common reasoning preamble from VC reader answers.

    VC readers sometimes prepend reasoning text like "Based on the search
    results..." before the actual answer.  This dilutes token-level F1
    even when the answer content is correct.  We strip known preamble
    patterns so the scorer evaluates the answer itself.
    """
    # Pattern: reasoning text followed by a quote-like answer or list
    # e.g. 'Based on the search results, I can see the games mentioned. Let me verify..."Detroit", ...'
    # Look for the last sentence-ending preamble before the actual answer
    # Repeatedly strip reasoning sentences from the start
    cleaned = s.strip()
    changed = True
    while changed:
        changed = False
        for pattern in [
            r'(?:Based on|According to|From|Looking at|After searching|After reviewing|I can see|I have found|The search|My search|Searching)(?:[^.!:]*)[.!:]\s*',
            r'Let me[^.!]*[.!]\s*',
            r'\[Previous reasoning compressed\]\s*',
        ]:
            m = re.match(pattern, cleaned, re.IGNORECASE | re.DOTALL)
            if m and len(cleaned[m.end():].strip()) > 5:
                cleaned = cleaned[m.end():]
                changed = True
    return cleaned.strip()


def normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punct, fix whitespace."""
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming."""
    pred_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    """F1 for multi-answer questions (comma-separated). Category 1."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    return sum(
        max((f1_score(p, gt) for p in predictions), default=0.0)
        for gt in ground_truths
    ) / len(ground_truths)


def score_question(prediction: str, question: LoCoMoQuestion) -> float:
    """Score a prediction against a LoCoMoQuestion. Returns F1 (0.0-1.0).

    Category-specific:
      1 (multi-hop): partial F1 over comma-separated sub-answers
      2 (temporal): standard F1
      3 (inference): standard F1, answer split on ';' (take first)
      4 (open-domain): standard F1
      5 (adversarial): binary — 1 if "not mentioned"/"no information available"
    """
    prediction = strip_reasoning_preamble(prediction)

    if question.category == 5:
        pred_lower = prediction.strip().lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0

    answer = question.answer
    if question.category == 3:
        answer = answer.split(";")[0].strip()

    if question.category == 1:
        return f1_multi_answer(prediction, answer)

    return f1_score(prediction, answer)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

_JUDGE_PROVIDERS = {
    "gemini": {
        "url_template": (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/{model}:generateContent?key={api_key}"
        ),
        "api_key_env": "GEMINI_API_KEY",
    },
    "anthropic": {
        "url_template": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openrouter": {
        "url_template": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


def judge_question(
    prediction: str,
    question: LoCoMoQuestion,
    *,
    model: str = "gemini-2.5-flash",
    provider: str = "gemini",
    api_key: str | None = None,
    conversation_date: str = "",
    evidence_session_date: str = "",
) -> dict:
    """Score a prediction using LLM-as-judge. Returns {correct, reasoning, cost}.

    Uses a lightweight model (Gemini Flash by default) for fast, cheap judging.
    """
    gold = question.answer
    if question.category == 5:
        gold = question.adversarial_answer or "Not mentioned in the conversation"

    prompt = _build_judge_prompt(
        question=question.question,
        gold=gold,
        prediction=prediction,
        category=question.category,
        conversation_date=evidence_session_date or conversation_date,
    )

    key = api_key or os.environ.get(
        _JUDGE_PROVIDERS.get(provider, {}).get("api_key_env", ""), ""
    )
    if not key:
        raise ValueError(f"No API key for judge provider '{provider}'")

    if provider == "gemini":
        return _judge_gemini(prompt, model, key, question.question_id)
    elif provider == "anthropic":
        return _judge_anthropic(prompt, model, key, question.question_id)
    elif provider == "openrouter":
        return _judge_openrouter(prompt, model, key, question.question_id)
    else:
        raise ValueError(f"Unsupported judge provider: {provider}")


def _judge_gemini(prompt: str, model: str, api_key: str, label: str) -> dict:
    """Call Gemini API for judging."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 64,
            "temperature": 0.0,
            "thinkingConfig": {"thinkingBudget": 1024},
        },
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers={"Content-Type": "application/json"}, json=payload)
        for _retry in range(3):
            if resp.status_code not in (429, 503):
                break
            time.sleep(min(5 * (2 ** _retry), 30))
            resp = client.post(url, headers={"Content-Type": "application/json"}, json=payload)

    if resp.status_code != 200:
        logger.error("Judge API error %d: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Judge API error {resp.status_code}")

    data = resp.json()
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    reasoning = "".join(p.get("text", "") for p in parts if "text" in p).strip()

    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    correct = "yes" in reasoning.lower().split("\n")[0].lower()

    logger.info("Judge [%s]: %s — %r", label, "CORRECT" if correct else "WRONG", reasoning)

    return {"correct": correct, "reasoning": reasoning, "input_tokens": input_tokens,
            "output_tokens": output_tokens}


def _judge_openrouter(prompt: str, model: str, api_key: str, label: str) -> dict:
    """Call OpenRouter API for judging (OpenAI-compatible format)."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        for _retry in range(3):
            if resp.status_code not in (429, 503):
                break
            time.sleep(min(5 * (2 ** _retry), 30))
            resp = client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error("Judge API error %d: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Judge API error {resp.status_code}")

    data = resp.json()
    choices = data.get("choices", [])
    reasoning = choices[0].get("message", {}).get("content", "").strip() if choices else ""

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    correct = "yes" in reasoning.lower().split("\n")[0].lower()

    logger.info("Judge [%s]: %s — %r", label, "CORRECT" if correct else "WRONG", reasoning)

    return {"correct": correct, "reasoning": reasoning, "input_tokens": input_tokens,
            "output_tokens": output_tokens}


def _judge_anthropic(prompt: str, model: str, api_key: str, label: str) -> dict:
    """Call Anthropic API for judging."""
    url = "https://api.anthropic.com/v1/messages"
    if api_key.startswith("sk-ant-oat"):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json",
        }
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    payload = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        for _retry in range(3):
            if resp.status_code not in (429, 503):
                break
            time.sleep(min(5 * (2 ** _retry), 30))
            resp = client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error("Judge API error %d: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Judge API error {resp.status_code}")

    data = resp.json()
    text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    reasoning = "\n".join(text_parts).strip()

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    correct = "yes" in reasoning.lower().split("\n")[0].lower()

    logger.info("Judge [%s]: %s — %r", label, "CORRECT" if correct else "WRONG", reasoning)

    return {"correct": correct, "reasoning": reasoning, "input_tokens": input_tokens,
            "output_tokens": output_tokens}
