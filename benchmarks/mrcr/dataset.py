"""Download, load, and select questions from the OpenAI MRCR dataset (HuggingFace)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# HuggingFace repo: openai/mrcr (MIT license)
HF_REPO = "openai/mrcr"
PARQUET_FILES = {
    2: "2needle/2needle_1.parquet",
    4: "4needle/4needle_1.parquet",
    8: "8needle/8needle_1.parquet",
}

# Context-length bins (upper bound in chars, label)
CONTEXT_BINS = [
    (8_000, "4k-8k"),
    (16_000, "8k-16k"),
    (32_000, "16k-32k"),
    (64_000, "32k-64k"),
    (128_000, "64k-128k"),
    (256_000, "128k-256k"),
    (512_000, "256k-512k"),
    (1_024_000, "512k-1m"),
    (float("inf"), "1m+"),
]


def _context_bin(n_chars: int) -> str:
    """Map character count to a context-length bin label."""
    for upper, label in CONTEXT_BINS:
        if n_chars <= upper:
            return label
    return "1m+"


@dataclass
class MRCRQuestion:
    question_id: str  # e.g. "2n_042"
    n_needles: int  # 2, 4, or 8
    desired_msg_index: int  # which needle instance to retrieve
    messages: list[dict]  # full conversation (parsed from prompt JSON)
    question_message: str  # the last user message (the retrieval request)
    answer: str  # gold answer text
    random_string: str  # string to prepend
    total_messages: int
    n_chars: int
    tokens_est: int  # n_chars // 4
    context_bin: str  # e.g. "128k-256k"


def download_dataset(force: bool = False) -> dict[int, Path]:
    """Download parquet files via huggingface_hub. Returns {needle_count: path}."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for MRCR dataset download. "
            "Install it with: pip install huggingface_hub"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}

    for n_needles, parquet_path in PARQUET_FILES.items():
        local_name = f"{n_needles}needle_1.parquet"
        dest = DATA_DIR / local_name

        if dest.exists() and not force:
            logger.info("MRCR %d-needle: already cached at %s", n_needles, dest)
            paths[n_needles] = dest
            continue

        logger.info("Downloading MRCR %d-needle from HuggingFace (%s)...", n_needles, parquet_path)
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=parquet_path,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        # hf_hub_download may place the file in a subdirectory; symlink/copy to flat location
        dl_path = Path(downloaded)
        if dl_path != dest:
            if dest.exists():
                dest.unlink()
            # Copy to flat location for simpler access
            import shutil
            shutil.copy2(dl_path, dest)

        logger.info("MRCR %d-needle: saved to %s", n_needles, dest)
        paths[n_needles] = dest

    return paths


def load_dataset(
    needle_counts: list[int] | None = None,
    data_dir: Path | None = None,
) -> list[MRCRQuestion]:
    """Load MRCR questions from parquet files.

    Parameters
    ----------
    needle_counts : list[int] | None
        Which needle variants to load (2, 4, 8). Default: all.
    data_dir : Path | None
        Override data directory. Default: benchmarks/mrcr/data/
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for MRCR dataset loading. "
            "Install it with: pip install pandas pyarrow"
        )

    d = data_dir or DATA_DIR
    counts = needle_counts or [2, 4, 8]
    questions: list[MRCRQuestion] = []

    for n_needles in counts:
        parquet_path = d / f"{n_needles}needle_1.parquet"
        if not parquet_path.exists():
            # Try subdirectory layout from hf_hub_download
            alt_path = d / f"{n_needles}needle" / f"{n_needles}needle_1.parquet"
            if alt_path.exists():
                parquet_path = alt_path
            else:
                logger.warning("MRCR %d-needle parquet not found at %s", n_needles, parquet_path)
                continue

        logger.info("Loading MRCR %d-needle from %s ...", n_needles, parquet_path)
        df = pd.read_parquet(parquet_path)

        for idx, row in df.iterrows():
            # Parse prompt field: JSON string → list of message dicts
            prompt_raw = row.get("prompt", "")
            if isinstance(prompt_raw, str):
                try:
                    all_messages = json.loads(prompt_raw)
                except json.JSONDecodeError:
                    logger.warning("MRCR %d-needle row %s: failed to parse prompt JSON", n_needles, idx)
                    continue
            elif isinstance(prompt_raw, list):
                all_messages = prompt_raw
            else:
                logger.warning("MRCR %d-needle row %s: unexpected prompt type %s", n_needles, idx, type(prompt_raw))
                continue

            if not all_messages:
                continue

            # Split: all messages except last user message = conversation history
            # Last user message = the retrieval question
            question_msg = ""
            history_messages = list(all_messages)
            # Walk backwards to find the last user message
            for i in range(len(all_messages) - 1, -1, -1):
                if all_messages[i].get("role") == "user":
                    question_msg = all_messages[i].get("content", "")
                    history_messages = all_messages[:i]
                    break

            n_chars = int(row.get("n_chars", 0))
            question_id = f"{n_needles}n_{idx:04d}"

            questions.append(MRCRQuestion(
                question_id=question_id,
                n_needles=n_needles,
                desired_msg_index=int(row.get("desired_msg_index", 0)),
                messages=history_messages,
                question_message=question_msg,
                answer=str(row.get("answer", "")),
                random_string=str(row.get("random_string_to_prepend", "")),
                total_messages=int(row.get("total_messages", len(all_messages))),
                n_chars=n_chars,
                tokens_est=n_chars // 4,
                context_bin=_context_bin(n_chars),
            ))

    logger.info("Loaded %d MRCR questions", len(questions))
    return questions


def select_questions(
    dataset: list[MRCRQuestion],
    count: int = 5,
    needle_counts: list[int] | None = None,
    bins: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> list[MRCRQuestion]:
    """Select a subset of questions with optional filtering.

    Parameters
    ----------
    count : int
        Maximum number of questions to return.
    needle_counts : list[int] | None
        Filter by needle count (2, 4, 8).
    bins : list[str] | None
        Filter by context bin (e.g. "128k-256k", "256k-512k").
    question_ids : list[str] | None
        Specific question IDs to select.
    """
    if question_ids:
        id_set = set(question_ids)
        selected = [q for q in dataset if q.question_id in id_set]
        if len(selected) < len(id_set):
            found = {q.question_id for q in selected}
            logger.warning("Question IDs not found: %s", id_set - found)
        return selected

    candidates = list(dataset)

    if needle_counts:
        candidates = [q for q in candidates if q.n_needles in needle_counts]

    if bins:
        bin_set = set(bins)
        candidates = [q for q in candidates if q.context_bin in bin_set]

    # Sort by context size (prefer smaller for smoke tests)
    candidates.sort(key=lambda q: q.n_chars)

    # Balance across needle counts if possible
    if not needle_counts:
        needle_groups: dict[int, list[MRCRQuestion]] = {}
        for q in candidates:
            needle_groups.setdefault(q.n_needles, []).append(q)

        selected: list[MRCRQuestion] = []
        selected_ids: set[str] = set()
        # Round-robin across needle counts
        group_iters = {k: iter(v) for k, v in sorted(needle_groups.items())}
        while len(selected) < count and group_iters:
            empty_keys = []
            for k, it in group_iters.items():
                if len(selected) >= count:
                    break
                try:
                    q = next(it)
                    if q.question_id not in selected_ids:
                        selected.append(q)
                        selected_ids.add(q.question_id)
                except StopIteration:
                    empty_keys.append(k)
            for k in empty_keys:
                del group_iters[k]
    else:
        selected = candidates[:count]

    logger.info(
        "Selected %d MRCR questions: %s",
        len(selected),
        [(q.question_id, f"{q.n_needles}n", q.context_bin) for q in selected],
    )
    return selected
