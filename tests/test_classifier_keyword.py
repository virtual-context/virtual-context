"""Tests for KeywordClassifier."""

import pytest

from virtual_context.classifiers.keyword import KeywordClassifier
from virtual_context.types import DomainDef


@pytest.fixture
def classifier():
    return KeywordClassifier()


@pytest.fixture
def domains():
    return [
        DomainDef(
            name="legal",
            keywords=["court", "filing", "attorney", "motion"],
            patterns=[r"\b\d{2}-cv-\d+"],
        ),
        DomainDef(
            name="medical",
            keywords=["insulin", "medication", "doctor", "glucose"],
        ),
        DomainDef(name="_general"),
    ]


@pytest.mark.asyncio
async def test_keyword_match(classifier, domains):
    await classifier.initialize(domains)
    results = await classifier.classify("The attorney filed a motion in court", domains)
    assert len(results) > 0
    assert results[0].domain == "legal"
    assert results[0].confidence >= 0.5


@pytest.mark.asyncio
async def test_regex_match(classifier, domains):
    await classifier.initialize(domains)
    results = await classifier.classify("Case number 24-cv-1234 was filed", domains)
    assert len(results) > 0
    legal = [r for r in results if r.domain == "legal"]
    assert legal[0].confidence == 0.9


@pytest.mark.asyncio
async def test_medical_match(classifier, domains):
    await classifier.initialize(domains)
    results = await classifier.classify("My doctor prescribed insulin for glucose control", domains)
    medical = [r for r in results if r.domain == "medical"]
    assert len(medical) > 0
    assert medical[0].confidence >= 0.5


@pytest.mark.asyncio
async def test_no_match(classifier, domains):
    await classifier.initialize(domains)
    results = await classifier.classify("The weather is nice today", domains)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multi_keyword_higher_confidence(classifier, domains):
    await classifier.initialize(domains)
    results1 = await classifier.classify("court", domains)
    results3 = await classifier.classify("court filing attorney motion", domains)
    if results1 and results3:
        assert results3[0].confidence > results1[0].confidence
