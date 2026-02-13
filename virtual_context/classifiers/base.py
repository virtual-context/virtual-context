"""Classifier ABC and ClassifierPipeline (ordered fallback chain)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ClassificationResult, DomainDef


class Classifier(ABC):
    """Base class for domain classifiers."""

    @abstractmethod
    async def classify(self, text: str, domains: list[DomainDef]) -> list[ClassificationResult]:
        """Return matching domains ordered by confidence descending. Empty = no opinion."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Classifier identifier (e.g. 'keyword', 'embedding', 'llm')."""

    async def initialize(self, domains: list[DomainDef]) -> None:
        """Optional: pre-compute domain representations (compile regex, compute embeddings)."""
        pass


class ClassifierPipeline:
    """Ordered fallback chain. First confident classifier wins."""

    def __init__(self, classifiers: list[Classifier], min_confidence: float = 0.3):
        self.classifiers = classifiers
        self.min_confidence = min_confidence

    async def initialize(self, domains: list[DomainDef]) -> None:
        for c in self.classifiers:
            await c.initialize(domains)

    async def classify(self, text: str, domains: list[DomainDef]) -> list[ClassificationResult]:
        """Run classifiers in order. First to return confident results wins.

        Guaranteed to return at least one result (_general as fallback).
        """
        for classifier in self.classifiers:
            results = await classifier.classify(text, domains)
            confident = [r for r in results if r.confidence >= self.min_confidence]
            if confident:
                return sorted(confident, key=lambda r: r.confidence, reverse=True)

        return [ClassificationResult(domain="_general", confidence=0.1, source="fallback")]
