"""File and directory loading utilities for conversation imports."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.types import Message


def load_from_path(
    path: Path,
    adapter: ExportAdapter,
) -> Iterator[tuple[str, list[Message]]]:
    """Load conversation(s) from a file or directory.

    Args:
        path: Path to a JSON file or directory containing JSON files.
        adapter: Export adapter to use for parsing.

    Yields:
        Tuples of (conversation_id, messages) for each successfully parsed file.
        Silently skips files that fail to parse.
    """
    if path.is_file():
        result = _load_single_file(path, adapter)
        if result:
            yield result
    else:
        for file_path in sorted(path.glob("*.json")):
            try:
                result = _load_single_file(file_path, adapter)
                if result:
                    yield result
            except (json.JSONDecodeError, OSError):
                continue


def _load_single_file(
    path: Path,
    adapter: ExportAdapter,
) -> tuple[str, list[Message]] | None:
    """Load a single conversation file.

    Args:
        path: Path to JSON file.
        adapter: Export adapter to use for parsing.

    Returns:
        Tuple of (conversation_id, messages) or None if parsing fails.
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    messages = adapter.extract_messages(data)
    conversation_id = adapter.extract_conversation_id(data)

    if not messages:
        return None

    return (conversation_id, messages)
