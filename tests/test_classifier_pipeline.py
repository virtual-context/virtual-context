"""Tests for ClassifierPipeline."""

import pytest

from virtual_context.classifiers.base import ClassifierPipeline
from virtual_context.classifiers.keyword import KeywordClassifier
from virtual_context.types import DomainDef


@pytest.fixture
def domains():
    return [
        DomainDef(name="legal", keywords=["court", "attorney"]),
        DomainDef(name="medical", keywords=["doctor", "insulin"]),
        DomainDef(name="_general"),
    ]


@pytest.mark.asyncio
async def test_pipeline_returns_result(domains):
    pipeline = ClassifierPipeline(
        classifiers=[KeywordClassifier()],
        min_confidence=0.3,
    )
    await pipeline.initialize(domains)
    results = await pipeline.classify("court attorney filing", domains)
    assert len(results) > 0
    assert results[0].domain == "legal"


@pytest.mark.asyncio
async def test_pipeline_fallback_to_general(domains):
    pipeline = ClassifierPipeline(
        classifiers=[KeywordClassifier()],
        min_confidence=0.3,
    )
    await pipeline.initialize(domains)
    results = await pipeline.classify("The weather is lovely today", domains)
    assert len(results) > 0
    assert results[0].domain == "_general"


@pytest.mark.asyncio
async def test_pipeline_confidence_filter(domains):
    pipeline = ClassifierPipeline(
        classifiers=[KeywordClassifier()],
        min_confidence=0.95,  # very high threshold
    )
    await pipeline.initialize(domains)
    # Single keyword won't reach 0.95
    results = await pipeline.classify("court", domains)
    assert results[0].domain == "_general"
