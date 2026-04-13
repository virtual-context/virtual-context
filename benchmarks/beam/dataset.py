"""Data loader for BEAM benchmark conversations and probing questions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


# Gold answer field names vary by category
_GOLD_ANSWER_FIELDS = (
    "answer",
    "ideal_response",
    "ideal_answer",
    "ideal_summary",
    "expected_compliance",
)


@dataclass
class BEAMQuestion:
    question_id: str
    question: str
    category: str
    rubric: list[str]
    ideal_response: str
    difficulty: str


@dataclass
class BEAMConversation:
    conv_id: str
    chat_size: str
    raw_data: list
    questions: list[BEAMQuestion] = field(default_factory=list)
    est_tokens: int = 0


def _extract_gold_answer(q: dict) -> str:
    """Extract gold answer from whichever field the category uses."""
    for f in _GOLD_ANSWER_FIELDS:
        if f in q and q[f]:
            val = q[f]
            return val if isinstance(val, str) else str(val)
    return ""


_SUFFIX_RE = re.compile(r"\s*->->\s*\d+,\d+\s*$")


def _strip_suffix(text: str) -> str:
    """Remove BEAM's ->-> N,N formatting artifact from message content."""
    return _SUFFIX_RE.sub("", text)


def flatten_messages(raw_data: list, chat_size: str) -> list[dict]:
    """Flatten BEAM chat.json into a list of {role, content, time_anchor} dicts.

    Handles both standard format (100K/500K/1M) and 10M format (extra plan nesting).
    """
    messages: list[dict] = []

    if chat_size == "10M":
        for plan in raw_data:
            # 10M: each item is a dict with one key mapping to a list of batches
            if isinstance(plan, dict):
                for batch_list in plan.values():
                    for batch in batch_list:
                        _extract_batch_messages(batch, messages)
            else:
                _extract_batch_messages(plan, messages)
    else:
        for batch in raw_data:
            _extract_batch_messages(batch, messages)

    return messages


def _extract_batch_messages(batch: dict, messages: list[dict]) -> None:
    """Extract messages from a single batch (has 'turns' key)."""
    for turn in batch.get("turns", []):
        for msg in turn:
            messages.append({
                "role": msg.get("role", "user"),
                "content": _strip_suffix(msg.get("content", "")),
                "time_anchor": msg.get("time_anchor", ""),
            })


def _load_probing_questions(pq_path: Path, conv_id: str) -> list[BEAMQuestion]:
    """Load probing questions from a conversation's probing_questions.json."""
    if not pq_path.exists():
        return []

    with open(pq_path) as f:
        data = json.load(f)

    questions: list[BEAMQuestion] = []
    for category, q_list in data.items():
        for idx, q in enumerate(q_list):
            questions.append(BEAMQuestion(
                question_id=f"{conv_id}_{category}_{idx}",
                question=q.get("question", ""),
                category=category,
                rubric=q.get("rubric", []),
                ideal_response=_extract_gold_answer(q),
                difficulty=q.get("difficulty", ""),
            ))

    return questions


def load_conversations(
    beam_root: str | Path,
    chat_size: str,
) -> list[BEAMConversation]:
    """Load all BEAM conversations for a given chat size.

    Args:
        beam_root: Path to BEAM repo root (contains chats/ directory).
        chat_size: One of "100K", "500K", "1M", "10M".

    Returns:
        List of BEAMConversation objects sorted by numeric conv_id.
    """
    chats_dir = Path(beam_root) / "chats" / chat_size
    if not chats_dir.exists():
        raise FileNotFoundError(f"BEAM chats directory not found: {chats_dir}")

    conversations: list[BEAMConversation] = []

    dirs = sorted(
        [d for d in chats_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    for conv_dir in dirs:
        chat_path = conv_dir / "chat.json"
        if not chat_path.exists():
            continue

        with open(chat_path) as f:
            raw_data = json.load(f)

        conv_id = f"{chat_size}_{conv_dir.name}"

        # Load probing questions
        pq_path = conv_dir / "probing_questions" / "probing_questions.json"
        questions = _load_probing_questions(pq_path, conv_id)

        # Estimate tokens from content length
        flat = flatten_messages(raw_data, chat_size)
        total_chars = sum(len(m["content"]) for m in flat)
        est_tokens = total_chars // 4

        conversations.append(BEAMConversation(
            conv_id=conv_id,
            chat_size=chat_size,
            raw_data=raw_data,
            questions=questions,
            est_tokens=est_tokens,
        ))

    return conversations
