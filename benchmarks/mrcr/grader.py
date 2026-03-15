"""Deterministic scoring for MRCR benchmark using SequenceMatcher."""

from __future__ import annotations

from difflib import SequenceMatcher


def grade(response: str, answer: str, random_string: str) -> float:
    """Score a response against the gold answer using MRCR's official grading logic.

    The model must prepend ``random_string`` to its response. If the prefix is
    missing the score is 0.0.  Otherwise both the response and answer have the
    prefix stripped and a SequenceMatcher ratio is computed (0.0-1.0).
    """
    if not response or not answer:
        return 0.0

    if not response.startswith(random_string):
        return 0.0

    response = response.removeprefix(random_string)
    answer = answer.removeprefix(random_string)

    return float(SequenceMatcher(None, response, answer).ratio())
