"""KeywordClassifier: regex + keyword matching, zero external deps."""

from __future__ import annotations

import re

from ..types import ClassificationResult, DomainDef
from .base import Classifier


class KeywordClassifier(Classifier):
    """Classify text by keyword and regex pattern matching.

    Confidence:
        - Regex pattern match: 0.9
        - Keyword match: 0.5 base + 0.05 per additional match (capped at 0.85)
    """

    def __init__(self) -> None:
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

    @property
    def name(self) -> str:
        return "keyword"

    async def initialize(self, domains: list[DomainDef]) -> None:
        self._compiled_patterns.clear()
        for domain in domains:
            if domain.patterns:
                self._compiled_patterns[domain.name] = [
                    re.compile(p, re.IGNORECASE) for p in domain.patterns
                ]

    async def classify(self, text: str, domains: list[DomainDef]) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        text_lower = text.lower()

        for domain in domains:
            if domain.name == "_general":
                continue

            confidence = 0.0

            # Check regex patterns (high confidence)
            patterns = self._compiled_patterns.get(domain.name, [])
            for pattern in patterns:
                if pattern.search(text):
                    confidence = max(confidence, 0.9)
                    break

            # Check keywords
            if domain.keywords:
                matches = sum(1 for kw in domain.keywords if kw.lower() in text_lower)
                if matches > 0:
                    kw_confidence = min(0.85, 0.5 + 0.05 * matches)
                    confidence = max(confidence, kw_confidence)

            if confidence > 0:
                results.append(
                    ClassificationResult(
                        domain=domain.name,
                        confidence=confidence,
                        source="keyword",
                    )
                )

        return sorted(results, key=lambda r: r.confidence, reverse=True)
