"""LLM-as-judge scoring matching LongMemEval's evaluate_qa.py prompts.

Judge templates are copied verbatim from the official LongMemEval repository:
https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
"""

from __future__ import annotations

import json as _json
import logging
import subprocess
import time

import httpx

from .auth import resolve_provider_token
from .cost import BudgetTracker
from .openai_codex import codex_simple_text_completion, extract_output_text, extract_usage

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Official LongMemEval judge prompt templates (verbatim)
# ---------------------------------------------------------------------------

JUDGE_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
    "response is still correct. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer."
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points "
    "in the rubric. The response is correct as long as it recalls and utilizes the "
    "user's personal information correctly."
    "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not."
    "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {hypothesis}"
    "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
)


def _get_judge_prompt(question_type: str) -> str:
    """Select the appropriate judge prompt template based on question type."""
    if "temporal" in question_type:
        return JUDGE_TEMPORAL
    elif "knowledge" in question_type:
        return JUDGE_KNOWLEDGE_UPDATE
    elif "preference" in question_type:
        return JUDGE_PREFERENCE
    elif "_abs" in question_type:
        return JUDGE_ABSTENTION
    else:
        return JUDGE_STANDARD


def judge_answer(
    question: str,
    answer: str,
    hypothesis: str,
    question_type: str,
    budget: BudgetTracker,
    label: str = "",
    model: str = "claude-haiku-4-5-20251001",
    provider: str = "anthropic",
    api_key: str | None = None,
    auth_mode: str = "auto",
) -> dict:
    """Score a hypothesis against the gold answer using LLM-as-judge.

    Returns dict with: correct (bool), explanation, cost.
    """
    template = _get_judge_prompt(question_type)
    prompt = template.format(question=question, answer=answer, hypothesis=hypothesis)

    if provider == "gemini-cli":
        result = subprocess.run(
            ["gemini", "-p", prompt, "--model", model, "--output-format", "json"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gemini CLI judge failed: {result.stderr[:300]}")
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        if json_start < 0:
            raise RuntimeError(f"gemini CLI judge returned no JSON: {stdout[:300]}")
        data = _json.loads(stdout[json_start:])
        explanation = data.get("response", "")
        input_tokens = 0
        output_tokens = 0
        for model_stats in data.get("stats", {}).get("models", {}).values():
            tokens = model_stats.get("tokens", {})
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("candidates", 0)
        entry = budget.record(label=f"judge:{label}", model=model,
                              input_tokens=input_tokens, output_tokens=output_tokens)
        correct = "yes" in explanation.lower().split("\n")[0].lower()
        logger.info("Judge [%s]: %s — $%.6f", label, "CORRECT" if correct else "WRONG", entry.cost_usd)
        return {"correct": correct, "explanation": explanation.strip(), "cost": entry.cost_usd}

    if provider == "gemini-oauth":
        from .baseline import _run_gemini_oauth
        oauth_result = _run_gemini_oauth(prompt, model, label, max_output_tokens=256, thinking_budget=128)
        explanation = oauth_result["hypothesis"]
        entry = budget.record(label=f"judge:{label}", model=model,
                              input_tokens=oauth_result["input_tokens"],
                              output_tokens=oauth_result["output_tokens"])
        correct = "yes" in explanation.lower().split("\n")[0].lower()
        logger.info("Judge [%s]: %s — $%.6f", label, "CORRECT" if correct else "WRONG", entry.cost_usd)
        return {"correct": correct, "explanation": explanation.strip(), "cost": entry.cost_usd}

    token = resolve_provider_token(provider, explicit_token=api_key, auth_mode=auth_mode)
    if not token:
        if provider in {"openai", "openai-codex", "openai_codex"}:
            raise ValueError(
                "OpenAI auth not found. Set OPENAI_API_KEY or OAuth token "
                "(OPENAI_OAUTH_ACCESS_TOKEN / OPENAI_OAUTH_TOKEN / OPENAI_ACCESS_TOKEN)."
            )
        if provider == "anthropic":
            raise ValueError("ANTHROPIC_API_KEY not set")
        if provider == "gemini":
            raise ValueError("GEMINI_API_KEY not set")
        raise ValueError(f"Unsupported provider or missing auth: {provider}")

    if provider == "anthropic":
        headers = {
            "x-api-key": token,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        url = API_URL
    elif provider == "openai":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_completion_tokens": 10,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        url = "https://api.openai.com/v1/chat/completions"
    elif provider in {"openai-codex", "openai_codex"}:
        headers = {}
        payload = {}
        url = ""
    elif provider == "gemini":
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 256,
                "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": 128},
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={token}"
    else:
        raise ValueError(f"Unsupported judge provider: {provider}")

    if provider in {"openai-codex", "openai_codex"}:
        data = codex_simple_text_completion(
            token=token,
            model=model,
            prompt=prompt,
            max_output_tokens=10,
            temperature=0.0,
            timeout_s=60.0,
        )
    else:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            # Retry on 429/503 with exponential backoff
            for _retry in range(5):
                if resp.status_code not in (429, 503):
                    break
                wait = min(10 * (2 ** _retry), 120)
                time.sleep(wait)
                resp = client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            logger.error("Judge API error %d: %s", resp.status_code, resp.text[:300])
            raise RuntimeError(f"Judge API error {resp.status_code}")

        data = resp.json()
    if provider == "anthropic":
        text_parts = [
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        explanation = "\n".join(text_parts)
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif provider == "openai":
        choices = data.get("choices", [])
        explanation = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    elif provider in {"openai-codex", "openai_codex"}:
        explanation = extract_output_text(data)
        input_tokens, output_tokens = extract_usage(data)
    else:  # gemini
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        explanation = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

    entry = budget.record(
        label=f"judge:{label}",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    correct = "yes" in explanation.lower().split("\n")[0].lower()

    logger.info("Judge [%s]: %s — $%.6f", label, "CORRECT" if correct else "WRONG", entry.cost_usd)

    return {
        "correct": correct,
        "explanation": explanation.strip(),
        "cost": entry.cost_usd,
    }
