"""Pure planning helpers for chronological canonical-turn resequencing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class CanonicalResequenceAssignment:
    canonical_turn_id: str
    namespace: str
    original_turn_group_number: int
    local_turn_number: int
    turn_group_number: int
    sort_key: float


def _value(row: object, name: str, default=""):
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        return row[name]  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        return getattr(row, name, default)


def _timestamp(row: object) -> datetime | None:
    for name in ("first_seen_at", "last_seen_at", "created_at", "updated_at"):
        raw = _value(row, name, "")
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _integer(row: object, name: str, default: int = -1) -> int:
    raw = _value(row, name, default)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _sort_key(row: object) -> float:
    raw = _value(row, "sort_key", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def plan_canonical_resequence(
    rows: list[object],
    *,
    owner_conversation_id: str,
) -> tuple[list[CanonicalResequenceAssignment], dict[tuple[str, int], int]]:
    """Rebuild logical pairs per origin, then globally order them by time.

    Each merged source originally had its own turn-number namespace.  The
    durable ``origin_conversation_id`` keeps those namespaces distinguishable.
    Within a namespace, physical sort order is the authoritative pairing
    fallback; across namespaces, the earliest durable row timestamp determines
    the unified guild chronology.
    """
    by_namespace: dict[str, list[object]] = {}
    for row in rows:
        origin = str(_value(row, "origin_conversation_id", "") or "").strip()
        namespace = origin or owner_conversation_id
        by_namespace.setdefault(namespace, []).append(row)

    logical_groups: list[tuple[datetime, str, int, list[object]]] = []
    for namespace, namespace_rows in by_namespace.items():
        ordered = sorted(
            namespace_rows,
            key=lambda row: (
                _sort_key(row),
                str(_value(row, "canonical_turn_id", "") or ""),
            ),
        )
        groups: list[list[object]] = []
        pending_user_group: list[object] | None = None
        for row in ordered:
            has_user = bool(str(_value(row, "user_content", "") or "").strip())
            has_assistant = bool(
                str(_value(row, "assistant_content", "") or "").strip()
            )
            if has_user and has_assistant:
                groups.append([row])
                pending_user_group = None
            elif has_user:
                pending_user_group = [row]
                groups.append(pending_user_group)
            elif has_assistant and pending_user_group is not None:
                pending_old_group = _integer(
                    pending_user_group[0], "turn_group_number", -1
                )
                assistant_old_group = _integer(
                    row, "turn_group_number", -1
                )
                if (
                    pending_old_group >= 0
                    and assistant_old_group >= 0
                    and pending_old_group != assistant_old_group
                ):
                    groups.append([row])
                else:
                    pending_user_group.append(row)
                pending_user_group = None
            elif not has_user and not has_assistant and pending_user_group is not None:
                pending_old_group = _integer(
                    pending_user_group[0], "turn_group_number", -1
                )
                empty_old_group = _integer(row, "turn_group_number", -1)
                if (
                    pending_old_group >= 0
                    and empty_old_group >= 0
                    and pending_old_group != empty_old_group
                ):
                    groups.append([row])
                    pending_user_group = None
                else:
                    # Provenance-only rows can sit between physical halves of
                    # the same old group. Keep those transparent to the pair.
                    pending_user_group.append(row)
            else:
                groups.append([row])
                pending_user_group = None

        group_timestamps: list[datetime | None] = []
        for group in groups:
            timestamps = [
                timestamp for row in group
                if (timestamp := _timestamp(row)) is not None
            ]
            group_timestamps.append(min(timestamps) if timestamps else None)
        for index, timestamp in enumerate(group_timestamps):
            if timestamp is not None:
                continue
            prior = next(
                (item for item in reversed(group_timestamps[:index]) if item is not None),
                None,
            )
            following = next(
                (item for item in group_timestamps[index + 1:] if item is not None),
                None,
            )
            group_timestamps[index] = (
                prior or following or datetime.min.replace(tzinfo=timezone.utc)
            )

        for local_turn_number, group in enumerate(groups):
            logical_groups.append((
                group_timestamps[local_turn_number],
                namespace,
                local_turn_number,
                group,
            ))

    logical_groups.sort(key=lambda item: (item[0], item[1], item[2]))
    assignments: list[CanonicalResequenceAssignment] = []
    old_to_global_candidates: dict[tuple[str, int], set[int]] = {}
    next_sort_slot = 1
    for global_turn_number, (_ts, namespace, local_turn_number, group) in enumerate(
        logical_groups
    ):
        for row in group:
            original_turn_group_number = _integer(
                row, "turn_group_number", -1
            )
            if original_turn_group_number >= 0:
                old_to_global_candidates.setdefault(
                    (namespace, original_turn_group_number), set()
                ).add(global_turn_number)
            assignments.append(CanonicalResequenceAssignment(
                canonical_turn_id=str(
                    _value(row, "canonical_turn_id", "") or ""
                ),
                namespace=namespace,
                original_turn_group_number=original_turn_group_number,
                local_turn_number=local_turn_number,
                turn_group_number=global_turn_number,
                sort_key=float(next_sort_slot * 1000),
            ))
            next_sort_slot += 1
    # Turn-scoped artifacts only carry the old numeric group plus origin.  A
    # reused old number cannot be guessed safely, so expose only unambiguous
    # mappings and let the storage layer fail closed if an artifact names one.
    artifact_mapping = {
        key: next(iter(candidates))
        for key, candidates in old_to_global_candidates.items()
        if len(candidates) == 1
    }
    return assignments, artifact_mapping
