"""BEAM-aligned LLM-as-judge evaluator.

Uses the exact scoring methodology from the BEAM paper:
- 0/0.5/1 scale per rubric item (not binary)
- One judge call per criterion (matching BEAM's evaluate_* functions)
- Kendall tau-b for event_ordering (matching BEAM's report_results.py)
- Semantic tolerance rules in the judge prompt
"""

from __future__ import annotations

import json
import logging
import math
import re
from difflib import SequenceMatcher
from typing import List, Tuple

from .dataset import BEAMQuestion

logger = logging.getLogger(__name__)

# Exact prompt from BEAM's src/prompts.py (unified_llm_judge_base_prompt)
_BEAM_JUDGE_PROMPT = """\
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


def judge_answer(
    question: BEAMQuestion,
    hypothesis: str,
    *,
    model: str = "gpt-4.1-mini",
    api_key: str = "",
    provider: str = "openai",
    api_url: str | None = None,
    event_ordering_mode: str = "full",
) -> dict:
    """Judge a model response against BEAM rubric criteria.

    Uses BEAM's exact methodology:
    - One LLM call per rubric criterion with the unified judge prompt
    - 0/0.5/1 scoring per criterion
    - For event_ordering: Kendall tau-b (normalized) as the primary score
    - For all others: mean of per-criterion scores

    Returns:
        {
            "score": float (0.0-1.0),
            "criteria_scores": list[dict],  # per-criterion {criterion, score, reason}
            "method": "llm_judge" | "kendall_tau",
        }
    """
    if not question.rubric:
        return {"score": 1.0, "criteria_scores": [], "method": "llm_judge"}

    # Event ordering uses Kendall tau-b as primary score (matching BEAM report_results.py)
    if question.category == "event_ordering":
        if event_ordering_mode == "fast":
            return _judge_event_ordering_fast(question, hypothesis)
        return _judge_event_ordering(question, hypothesis,
                                     model=model, api_key=api_key,
                                     provider=provider, api_url=api_url)

    # All other categories: LLM judge per rubric item, 0/0.5/1 scale
    return _judge_llm_rubric(question, hypothesis,
                             model=model, api_key=api_key,
                             provider=provider, api_url=api_url)


def _judge_llm_rubric(
    question: BEAMQuestion,
    hypothesis: str,
    *,
    model: str,
    api_key: str,
    provider: str,
    api_url: str | None,
) -> dict:
    """Judge using BEAM's unified LLM judge prompt, one call per criterion."""
    criteria_scores: list[dict] = []

    for criterion in question.rubric:
        prompt = _BEAM_JUDGE_PROMPT \
            .replace("<question>", question.question) \
            .replace("<rubric_item>", criterion) \
            .replace("<llm_response>", hypothesis)

        try:
            response_text = _call_judge_llm(prompt, model=model, api_key=api_key,
                                            provider=provider, api_url=api_url)
            parsed = _parse_score_response(response_text)
            criteria_scores.append({
                "criterion": criterion,
                "score": parsed["score"],
                "reason": parsed.get("reason", ""),
            })
        except Exception as e:
            logger.warning("Judge call failed for criterion %r: %s", criterion[:60], e)
            criteria_scores.append({
                "criterion": criterion,
                "score": 0.0,
                "reason": f"Judge error: {e}",
            })

    total = sum(c["score"] for c in criteria_scores)
    score = total / len(criteria_scores) if criteria_scores else 0.0

    # Also produce legacy fields for backward compat with run_beam summary
    criteria_met = [c["criterion"] for c in criteria_scores if c["score"] >= 1.0]
    criteria_missed = [c["criterion"] for c in criteria_scores if c["score"] == 0.0]

    return {
        "score": score,
        "criteria_scores": criteria_scores,
        "criteria_met": criteria_met,
        "criteria_missed": criteria_missed,
        "method": "llm_judge",
    }


_FAST_ALIGN_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "with",
}


def _normalize_alignment_text(text: str) -> tuple[str, list[str]]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    tokens = [
        token for token in cleaned.split()
        if len(token) >= 3 and token not in _FAST_ALIGN_STOPWORDS
    ]
    return cleaned, tokens


def _fast_alignment_score(first: str, second: str) -> float:
    first_norm, first_tokens = _normalize_alignment_text(first)
    second_norm, second_tokens = _normalize_alignment_text(second)
    if not first_norm or not second_norm:
        return 0.0

    first_set = set(first_tokens)
    second_set = set(second_tokens)
    overlap = len(first_set & second_set)
    union = len(first_set | second_set)
    jaccard = overlap / union if union else 0.0
    containment = overlap / min(len(first_set), len(second_set)) if first_set and second_set else 0.0
    ratio = SequenceMatcher(None, first_norm, second_norm).ratio()

    return 0.5 * containment + 0.3 * jaccard + 0.2 * ratio


def _align_without_llm(
    reference: List[str],
    system: List[str],
    *,
    threshold: float = 0.33,
) -> tuple[List[str], List[str], list[dict]]:
    """Cheap lexical alignment for event ordering iteration.

    We greedily match each system item to the best unused reference item using a
    similarity score built from token overlap and sequence similarity.
    """
    scored_pairs: list[tuple[float, int, int]] = []
    for sys_idx, system_item in enumerate(system):
        for ref_idx, reference_item in enumerate(reference):
            score = _fast_alignment_score(reference_item, system_item)
            if score >= threshold:
                scored_pairs.append((score, sys_idx, ref_idx))

    scored_pairs.sort(reverse=True)

    used_system: set[int] = set()
    used_reference: set[int] = set()
    replacements: dict[int, int] = {}
    alignments: list[dict] = []

    for score, sys_idx, ref_idx in scored_pairs:
        if sys_idx in used_system or ref_idx in used_reference:
            continue
        used_system.add(sys_idx)
        used_reference.add(ref_idx)
        replacements[sys_idx] = ref_idx
        alignments.append({
            "system_index": sys_idx,
            "reference_index": ref_idx,
            "system_item": system[sys_idx],
            "reference_item": reference[ref_idx],
            "score": round(score, 3),
        })

    system_out: list[str] = []
    for sys_idx, system_item in enumerate(system):
        ref_idx = replacements.get(sys_idx)
        system_out.append(reference[ref_idx] if ref_idx is not None else system_item)

    return reference, system_out, alignments


def _judge_event_ordering_fast(
    question: BEAMQuestion,
    hypothesis: str,
) -> dict:
    """Cheap event-ordering judge for iteration.

    Uses lexical alignment instead of pairwise LLM equivalence, then computes the
    same Kendall tau-b based ordering score.
    """
    from scipy.stats import kendalltau

    reference_list = question.rubric
    system_list = [line.strip() for line in hypothesis.split("\n") if line.strip()]

    reference_canon, system_canon, alignments = _align_without_llm(reference_list, system_list)

    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        ranks = {item: i + 1 for i, item in enumerate(seq)}
        return [ranks.get(item, tie_rank) for item in union]

    tau_b, _ = kendalltau(to_rank(reference_canon), to_rank(system_canon),
                          variant="b", method="auto")
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None and not math.isnan(tau_b) else 0.0

    matched_reference = set(system_canon) & set(reference_canon)
    criteria_scores = []
    criteria_met = []
    criteria_missed = []
    for criterion in reference_list:
        if criterion in matched_reference:
            criteria_scores.append({
                "criterion": criterion,
                "score": 1.0,
                "reason": "Matched by fast lexical alignment.",
            })
            criteria_met.append(criterion)
        else:
            criteria_scores.append({
                "criterion": criterion,
                "score": 0.0,
                "reason": "No aligned system item matched this reference event.",
            })
            criteria_missed.append(criterion)

    return {
        "score": tau_b_norm,
        "tau_norm": tau_b_norm,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "criteria_scores": criteria_scores,
        "criteria_met": criteria_met,
        "criteria_missed": criteria_missed,
        "alignments": alignments,
        "method": "kendall_tau_fast",
    }


def _judge_event_ordering(
    question: BEAMQuestion,
    hypothesis: str,
    *,
    model: str,
    api_key: str,
    provider: str,
    api_url: str | None,
) -> dict:
    """Event ordering: Kendall tau-b as primary score (matching BEAM report_results.py).

    The rubric items are the reference-ordered list of events.
    The hypothesis is split by newlines to get the system's ordering.
    Kendall tau-b rank correlation is computed between the two orderings.
    """
    from scipy.stats import kendalltau

    reference_list = question.rubric
    system_list = [line.strip() for line in hypothesis.split("\n") if line.strip()]

    # Semantic alignment: match system items to reference items via LLM equivalence
    reference_canon, system_canon = _align_with_llm(
        reference_list, system_list,
        model=model, api_key=api_key, provider=provider, api_url=api_url,
    )

    # Compute set-based precision/recall/F1
    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # Kendall tau-b on the union ordering
    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference_canon), to_rank(system_canon),
                          variant="b", method="auto")
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None and not math.isnan(tau_b) else 0.0

    # Primary score is tau_b_norm (matching BEAM's report_results.py line 46)
    final_score = tau_b_norm

    # Also run LLM judge per rubric item for diagnostic detail
    llm_result = _judge_llm_rubric(question, hypothesis,
                                   model=model, api_key=api_key,
                                   provider=provider, api_url=api_url)

    return {
        "score": final_score,
        "tau_norm": tau_b_norm,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "llm_judge_score": llm_result["score"],
        "criteria_scores": llm_result["criteria_scores"],
        "criteria_met": llm_result.get("criteria_met", []),
        "criteria_missed": llm_result.get("criteria_missed", []),
        "method": "kendall_tau",
    }


def _align_with_llm(
    reference: List[str],
    system: List[str],
    *,
    model: str,
    api_key: str,
    provider: str,
    api_url: str | None,
) -> Tuple[List[str], List[str]]:
    """Replace each system item by its matched reference item if LLM says they're equivalent.

    Mirrors BEAM's align_with_llm() from compute_metrics.py.
    """
    used: set[int] = set()
    system_out: list[str] = []

    for s in system:
        matched_index = None
        for idx, r in enumerate(reference):
            if idx in used:
                continue
            if _llm_equivalence(r, s, model=model, api_key=api_key,
                                provider=provider, api_url=api_url):
                matched_index = idx
                break

        if matched_index is not None:
            system_out.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_out.append(s)

    return reference, system_out


def _llm_equivalence(
    first: str,
    second: str,
    *,
    model: str,
    api_key: str,
    provider: str,
    api_url: str | None,
) -> bool:
    """Check if two snippets describe the same event/fact (mirrors BEAM's llm_equivalence)."""
    prompt = (
        "You are a binary classifier.\n"
        "If the TWO snippets describe the SAME event/fact, reply **YES**\n"
        "Otherwise reply **NO**. No extra words.\n"
        "DO NOT provide any explanation.\n\n"
        f"First snippet: {first}\n\n"
        f"Second snippet: {second}"
    )

    try:
        response = _call_judge_llm(prompt, model=model, api_key=api_key,
                                   provider=provider, api_url=api_url)
        return "yes" in response.lower()
    except Exception as e:
        logger.warning("LLM equivalence check failed: %s", e)
        return False


def _parse_score_response(response_text: str) -> dict:
    """Parse a judge response expecting {"score": 0/0.5/1, "reason": "..."}."""
    cleaned = response_text.strip()

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "score" in parsed:
            # Clamp to valid values
            raw = float(parsed["score"])
            if raw >= 0.75:
                score = 1.0
            elif raw >= 0.25:
                score = 0.5
            else:
                score = 0.0
            return {"score": score, "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to find JSON in response
    match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', response_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            raw = float(parsed["score"])
            if raw >= 0.75:
                score = 1.0
            elif raw >= 0.25:
                score = 0.5
            else:
                score = 0.0
            return {"score": score, "reason": parsed.get("reason", "")}
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse judge response, defaulting to 0.0: %s", response_text[:200])
    return {"score": 0.0, "reason": "Failed to parse judge response"}


def _call_judge_llm(
    prompt: str,
    *,
    model: str,
    api_key: str,
    provider: str,
    api_url: str | None,
) -> str:
    """Call the judge LLM via OpenAI-compatible or Anthropic API."""
    import urllib.request

    if provider == "openai":
        url = api_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    elif provider == "anthropic":
        url = api_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    elif provider == "openrouter":
        url = api_url or "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    else:
        raise ValueError(f"Unsupported judge provider: {provider}")

    if provider == "anthropic":
        body = {
            "model": model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        body = {
            "model": model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())

    if provider == "anthropic":
        return result["content"][0]["text"]
    else:
        return result["choices"][0]["message"]["content"]
