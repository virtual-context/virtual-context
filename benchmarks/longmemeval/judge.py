"""LLM-as-judge scoring matching LongMemEval's evaluate_qa.py prompts.

Judge templates are copied verbatim from the official LongMemEval repository:
https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
"""

from __future__ import annotations

import logging
import os

import httpx

from .cost import BudgetTracker

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
    api_key: str | None = None,
) -> dict:
    """Score a hypothesis against the gold answer using LLM-as-judge.

    Returns dict with: correct (bool), explanation, cost.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    template = _get_judge_prompt(question_type)
    prompt = template.format(question=question, answer=answer, hypothesis=hypothesis)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 10,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error("Judge API error %d: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Judge API error {resp.status_code}")

    data = resp.json()
    text_parts = [
        block["text"] for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    explanation = "\n".join(text_parts)

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

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
