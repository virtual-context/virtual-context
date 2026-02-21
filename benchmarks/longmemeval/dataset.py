"""Download, load, and select questions from the LongMemEval dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
DATA_DIR = Path(__file__).parent / "data"

# Question types in LongMemEval (_abs suffix = abstention variant)
CATEGORIES = [
    "single-session-user",         # IE: info extraction from user messages
    "single-session-assistant",    # IE: info extraction from assistant messages
    "multi-session",               # MR: cross-conversation reasoning
    "temporal-reasoning",          # TR: time-based inference
    "knowledge-update",            # KU: tracking information changes
]


@dataclass
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: list[list[dict]]  # sessions → turns → {role, content}
    haystack_dates: list[str]
    answer_session_ids: list[str]
    haystack_tokens_est: int = 0


def download_dataset(force: bool = False) -> Path:
    """Download longmemeval_s_cleaned.json if not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "longmemeval_s_cleaned.json"

    if dest.exists() and not force:
        logger.info("Dataset already cached at %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    logger.info("Downloading LongMemEval dataset from HuggingFace...")
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        with client.stream("GET", DATASET_URL) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info("  %.0f%% (%.1f / %.1f MB)", pct, downloaded / 1e6, total / 1e6)

    logger.info("Downloaded to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load_dataset(path: Path | None = None) -> list[LongMemEvalQuestion]:
    """Load all questions from the dataset JSON."""
    if path is None:
        path = DATA_DIR / "longmemeval_s_cleaned.json"

    if not path.exists():
        path = download_dataset()

    logger.info("Loading dataset from %s ...", path)
    raw = json.loads(path.read_text())

    questions: list[LongMemEvalQuestion] = []
    for item in raw:
        haystack_sessions = item.get("haystack_sessions", [])
        est_tokens = _estimate_tokens(haystack_sessions)

        q = LongMemEvalQuestion(
            question_id=str(item.get("question_id", item.get("id", ""))),
            question_type=item.get("question_type", item.get("type", "")),
            question=item.get("question", ""),
            answer=str(item.get("answer", "")),
            question_date=item.get("question_date", item.get("date", "")),
            haystack_sessions=haystack_sessions,
            haystack_dates=item.get("haystack_dates", item.get("dates", [])),
            answer_session_ids=[str(x) for x in item.get("answer_session_id", item.get("answer_session_ids", []))],
            haystack_tokens_est=est_tokens,
        )
        questions.append(q)

    logger.info("Loaded %d questions", len(questions))
    return questions


def _estimate_tokens(sessions: list[list[dict]]) -> int:
    """Rough token estimate: chars / 4."""
    total_chars = 0
    for session in sessions:
        for turn in session:
            content = turn.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
    return total_chars // 4


def select_questions(
    dataset: list[LongMemEvalQuestion],
    count: int = 5,
    categories: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> list[LongMemEvalQuestion]:
    """Select a balanced subset of questions.

    Strategy: 1 per category, prefer shorter haystacks, skip _abs (abstention) for smoke test.
    """
    if question_ids:
        id_set = set(question_ids)
        selected = [q for q in dataset if q.question_id in id_set]
        if len(selected) < len(id_set):
            found = {q.question_id for q in selected}
            missing = id_set - found
            logger.warning("Question IDs not found: %s", missing)
        return selected

    target_categories = categories or CATEGORIES
    # Skip abstention variants for smoke test
    candidates = [q for q in dataset if not q.question_type.endswith("_abs")]

    # Group by category
    by_cat: dict[str, list[LongMemEvalQuestion]] = {}
    for q in candidates:
        cat = q.question_type
        by_cat.setdefault(cat, []).append(q)

    # Sort each category by haystack size (prefer shorter)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda q: q.haystack_tokens_est)

    selected: list[LongMemEvalQuestion] = []

    # Pick 1 from each target category
    for cat in target_categories:
        if cat in by_cat and by_cat[cat]:
            selected.append(by_cat[cat].pop(0))
            if len(selected) >= count:
                break

    # If we need more, fill from remaining (shortest first across all cats)
    if len(selected) < count:
        remaining = []
        for cat_list in by_cat.values():
            remaining.extend(cat_list)
        remaining.sort(key=lambda q: q.haystack_tokens_est)
        selected_ids = {q.question_id for q in selected}
        for q in remaining:
            if q.question_id not in selected_ids:
                selected.append(q)
                if len(selected) >= count:
                    break

    logger.info(
        "Selected %d questions: %s",
        len(selected),
        [(q.question_id, q.question_type, f"~{q.haystack_tokens_est}t") for q in selected],
    )
    return selected
