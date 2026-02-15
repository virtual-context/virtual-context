"""Data layer for TUI chat state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..types import AssembledContext, CompactionReport


@dataclass
class TurnRecord:
    """Snapshot of one user-assistant exchange."""

    turn_number: int
    user_message: str
    assistant_message: str
    assembled: AssembledContext
    compaction: CompactionReport | None = None
    tags: list[str] = field(default_factory=list)
    primary_tag: str = "_general"
    broad: bool = False
    temporal: bool = False
    input_tokens: int = 0
    turns_in_payload: int = 0
    api_payload: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_export_dict(self) -> dict:
        """Serializable dict for JSON export."""
        d = {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "tags": self.tags,
            "primary_tag": self.primary_tag,
            "broad": self.broad,
            "temporal": self.temporal,
            "input_tokens": self.input_tokens,
            "turns_in_payload": self.turns_in_payload,
            "api_payload": self.api_payload,
        }
        if self.timing:
            d["timing_ms"] = self.timing
        return d


def save_turn(turn: TurnRecord, directory: str = ".") -> Path:
    """Save a single turn to vc-turn-{N}.json. Returns the file path."""
    path = Path(directory) / f"vc-turn-{turn.turn_number}.json"
    path.write_text(json.dumps(turn.to_export_dict(), indent=2, ensure_ascii=False))
    return path


def save_session(turns: list[TurnRecord], directory: str = ".") -> Path:
    """Save all turns to vc-session.json. Returns the file path."""
    path = Path(directory) / "vc-session.json"
    data = {
        "total_turns": len(turns),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "turns": [t.to_export_dict() for t in turns],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_replay_prompts(path: str | Path) -> list[str]:
    """Load prompts from a session JSON or a plain-text file.

    Supports two formats:
    - **vc-session.json**: extracts ``user_message`` from each turn
    - **Plain text**: one prompt per line (blank lines ignored)

    Returns a list of user prompt strings.
    """
    p = Path(path)
    text = p.read_text()

    # Try JSON session format first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "turns" in data:
            return [t["user_message"] for t in data["turns"] if t.get("user_message")]
        if isinstance(data, list):
            # Bare list of strings
            return [item if isinstance(item, str) else item.get("user_message", "") for item in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fall back to plain text — one prompt per non-blank line
    return [line.strip() for line in text.splitlines() if line.strip()]
