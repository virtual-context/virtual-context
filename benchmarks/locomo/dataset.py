"""LocOMo benchmark dataset loader and dataclasses."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_FILE = Path("/Users/yursilkidwai/projects/locomo/data/locomo10.json")

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "inference",
    4: "open-domain",
    5: "adversarial",
}


@dataclass
class LoCoMoQuestion:
    """A single QA pair from a LocOMo conversation."""

    question_id: str  # "{conv_id}_q{idx}"
    conv_id: str
    qa_index: int
    question: str
    answer: str  # Gold answer (coerced to str; empty for cat 5)
    adversarial_answer: str  # Cat 5 only, else ""
    evidence: list[str] = field(default_factory=list)  # ["D1:3"]
    category: int = 0


@dataclass
class LoCoMoSession:
    """A single session (dialog) within a conversation."""

    session_num: int
    date_time: str
    turns: list[dict] = field(default_factory=list)


@dataclass
class LoCoMoConversation:
    """One of the 10 LocOMo conversations."""

    conv_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[LoCoMoSession] = field(default_factory=list)
    questions: list[LoCoMoQuestion] = field(default_factory=list)
    total_turns: int = 0
    est_tokens: int = 0


def load_dataset(path: Path | None = None) -> list[LoCoMoConversation]:
    """Load the LocOMo dataset from locomo10.json."""
    p = path or DATA_FILE
    if not p.exists():
        raise FileNotFoundError(f"LocOMo data not found at {p}")

    logger.info("Loading LocOMo dataset from %s ...", p)
    with open(p) as f:
        raw = json.load(f)

    conversations: list[LoCoMoConversation] = []
    for entry in raw:
        conv_id = entry["sample_id"]
        conv_data = entry["conversation"]
        speaker_a = conv_data["speaker_a"]
        speaker_b = conv_data["speaker_b"]

        # Extract sessions sorted numerically
        session_keys = sorted(
            [k for k in conv_data if re.match(r"^session_\d+$", k)],
            key=lambda k: int(k.split("_")[1]),
        )

        sessions: list[LoCoMoSession] = []
        total_turns = 0
        total_chars = 0
        for sk in session_keys:
            num = int(sk.split("_")[1])
            dt_key = f"{sk}_date_time"
            date_time = conv_data.get(dt_key, "")
            turns = conv_data[sk]
            total_turns += len(turns)
            for t in turns:
                total_chars += len(t.get("text", ""))
            sessions.append(LoCoMoSession(
                session_num=num,
                date_time=date_time,
                turns=turns,
            ))

        # Build questions
        questions: list[LoCoMoQuestion] = []
        for idx, qa in enumerate(entry.get("qa", [])):
            cat = qa.get("category", 0)
            answer = str(qa["answer"]) if "answer" in qa else ""
            adversarial = str(qa.get("adversarial_answer", ""))
            questions.append(LoCoMoQuestion(
                question_id=f"{conv_id}_q{idx}",
                conv_id=conv_id,
                qa_index=idx,
                question=qa["question"],
                answer=answer,
                adversarial_answer=adversarial,
                evidence=qa.get("evidence", []),
                category=cat,
            ))

        conversations.append(LoCoMoConversation(
            conv_id=conv_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            questions=questions,
            total_turns=total_turns,
            est_tokens=total_chars // 4,
        ))

    logger.info(
        "Loaded %d conversations, %d total QA pairs",
        len(conversations),
        sum(len(c.questions) for c in conversations),
    )
    return conversations


def select_conversations(
    dataset: list[LoCoMoConversation],
    conv_ids: list[str] | None = None,
    count: int | None = None,
) -> list[LoCoMoConversation]:
    """Select conversations by ID or count."""
    if conv_ids:
        selected = [c for c in dataset if c.conv_id in conv_ids]
        if not selected:
            raise ValueError(f"No conversations found for IDs: {conv_ids}")
        return selected
    if count is not None:
        return dataset[:count]
    return dataset


def select_questions(
    conversation: LoCoMoConversation,
    categories: list[int] | None = None,
    count: int | None = None,
    question_ids: list[str] | None = None,
) -> list[LoCoMoQuestion]:
    """Filter questions within a conversation."""
    qs = conversation.questions
    if question_ids:
        qs = [q for q in qs if q.question_id in question_ids]
    if categories:
        qs = [q for q in qs if q.category in categories]
    if count is not None:
        qs = qs[:count]
    return qs
