"""Shared fixtures for virtual-context tests."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from virtual_context.config import load_config
from virtual_context.types import (
    KeywordTagConfig,
    Message,
    TagGeneratorConfig,
    TagResult,
    VirtualContextConfig,
)


@pytest.fixture
def sample_tag_config() -> TagGeneratorConfig:
    return TagGeneratorConfig(
        type="keyword",
        max_tags=5,
        min_tags=1,
        keyword_fallback=KeywordTagConfig(
            tag_keywords={
                "legal": ["court", "filing", "motion", "attorney", "legal", "case", "judge"],
                "medical": ["insulin", "medication", "doctor", "lab", "blood", "glucose"],
            },
            tag_patterns={
                "legal": [r"\bN\.?J\.?S\.?A\.?\b", r"\b\d{2}-cv-\d+"],
            },
        ),
    )


@pytest.fixture
def sample_config(sample_tag_config) -> VirtualContextConfig:
    return load_config(config_dict={
        "context_window": 10000,
        "tag_generator": {
            "type": "keyword",
            "keyword_fallback": {
                "tag_keywords": {
                    "legal": ["court", "filing", "motion", "attorney", "legal", "case", "judge"],
                    "medical": ["insulin", "medication", "doctor", "lab", "blood", "glucose"],
                },
                "tag_patterns": {
                    "legal": [r"\bN\.?J\.?S\.?A\.?\b"],
                },
            },
        },
        "tag_rules": [
            {"match": "legal*", "priority": 9},
            {"match": "medical*", "priority": 8},
            {"match": "*", "priority": 5},
        ],
        "compaction": {
            "soft_threshold": 0.70,
            "hard_threshold": 0.85,
            "protected_recent_turns": 2,
        },
    })


@pytest.fixture
def ts() -> datetime:
    return datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def legal_messages(ts) -> list[Message]:
    return [
        Message(role="user", content="What's the deadline for the court filing in case 24-cv-1234?", timestamp=ts),
        Message(role="assistant", content="The filing deadline for case 24-cv-1234 is January 30th. The motion must be submitted to the court by 5pm.", timestamp=ts + timedelta(seconds=30)),
        Message(role="user", content="Has the attorney reviewed the settlement offer?", timestamp=ts + timedelta(minutes=2)),
        Message(role="assistant", content="Yes, the attorney reviewed the settlement offer and recommends we counter at $50,000.", timestamp=ts + timedelta(minutes=2, seconds=30)),
    ]


@pytest.fixture
def medical_messages(ts) -> list[Message]:
    base = ts + timedelta(minutes=10)
    return [
        Message(role="user", content="My blood glucose was 180 this morning. Should I adjust my insulin?", timestamp=base),
        Message(role="assistant", content="A reading of 180 is above target. Consider adjusting your insulin dosage. Check with your doctor about increasing by 1 unit.", timestamp=base + timedelta(seconds=30)),
        Message(role="user", content="The lab results from last week showed elevated glucose levels too.", timestamp=base + timedelta(minutes=2)),
        Message(role="assistant", content="Consistently elevated glucose warrants a medication review. Schedule an appointment with your endocrinologist.", timestamp=base + timedelta(minutes=2, seconds=30)),
    ]


@pytest.fixture
def mixed_messages(legal_messages, medical_messages) -> list[Message]:
    return legal_messages + medical_messages


@pytest.fixture
def tmp_store_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def tmp_sqlite_db(tmp_store_dir):
    return tmp_store_dir / "test_store.db"


class FakeChatProvider:
    """Mock chat provider that yields canned responses (no API calls)."""

    def __init__(self, responses: list[str] | None = None):
        self.api_key = "fake-key"
        self.model = "fake-model"
        self._responses = responses or ["Hello! I'm a test assistant."]
        self._call_count = 0

    def stream_message(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ):
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class MockLLMProvider:
    """Mock LLM provider for testing compaction."""

    def __init__(self, response: str | None = None):
        self.calls: list[dict] = []
        self.response = response or (
            '{"summary": "Test summary", "entities": ["entity1"], '
            '"key_decisions": ["decision1"], "action_items": [], '
            '"date_references": [], "refined_tags": ["test-tag"]}'
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.response


class MockTagGenerator:
    """Mock tag generator that returns deterministic results."""

    def __init__(self, default_tag: str = "test-tag", default_tags: list[str] | None = None):
        self.default_tag = default_tag
        self.default_tags = default_tags or [default_tag]
        self.calls: list[str] = []
        self._overrides: dict[str, TagResult] = {}

    def set_override(self, keyword: str, result: TagResult) -> None:
        """Set a tag result override for text containing keyword."""
        self._overrides[keyword] = result

    def generate_tags(self, text: str, existing_tags: list[str] | None = None) -> TagResult:
        self.calls.append(text)
        # Check overrides
        for keyword, result in self._overrides.items():
            if keyword.lower() in text.lower():
                return result
        return TagResult(
            tags=list(self.default_tags),
            primary=self.default_tag,
            source="mock",
        )
