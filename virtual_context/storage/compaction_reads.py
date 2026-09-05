"""Read pending canonical groups without hydrating compacted archive bodies."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import logging
from itertools import islice
import uuid

from ..core.canonical_turns import STRIP_WHITESPACE

logger = logging.getLogger(__name__)


@contextmanager
def _scalar_cursor(store, conn, query, params):
    cursor = (conn.cursor(name="vc_pending_" + uuid.uuid4().hex)
              if store._relational_dialect == "postgres" else conn.cursor())
    try:
        cursor.execute(query, params)
        def rows():
            while batch := cursor.fetchmany(200):
                yield from batch
        yield rows()
    finally:
        cursor.close()


def _group_mode(conn, p, conversation_id):
    row = conn.execute(f"""SELECT MIN(COALESCE(turn_group_number,-1)) AS smallest,
        MAX(COALESCE(turn_group_number,-1)) AS largest FROM canonical_turns
        WHERE conversation_id={p}""", (conversation_id,)).fetchone()
    return row["smallest"], row["largest"]


def _backfill_all_legacy_groups(store, conversation_id):
    """Preserve the established lazy repair without loading any text bodies."""
    p = store._placeholder
    with store._relational_connection() as conn:
        _smallest, largest = _group_mode(conn, p, conversation_id)
    if largest is None or largest >= 0:
        return
    try:
        with store._relational_connection(write=True, scope=conversation_id) as conn:
            _smallest, largest = _group_mode(conn, p, conversation_id)
            if largest is None or largest >= 0:
                return
            trim = "btrim" if store._relational_dialect == "postgres" else "trim"
            query = f"""SELECT canonical_turn_id,
                CASE WHEN {trim}(COALESCE(user_content,''),{p})<>'' THEN 1 ELSE 0 END AS has_user,
                CASE WHEN {trim}(COALESCE(assistant_content,''),{p})<>'' THEN 1 ELSE 0 END AS has_assistant
                FROM canonical_turns WHERE conversation_id={p}
                ORDER BY sort_key,canonical_turn_id"""
            current, pending = -1, -1
            with _scalar_cursor(store, conn, query, (STRIP_WHITESPACE, STRIP_WHITESPACE, conversation_id)) as rows:
                for row in rows:
                    user, assistant = row["has_user"], row["has_assistant"]
                    if assistant and not user and pending >= 0:
                        group, pending = pending, -1
                    else:
                        current += 1
                        group = current
                        pending = current if user and not assistant else -1
                    conn.execute(f"""UPDATE canonical_turns SET turn_group_number={p}
                        WHERE conversation_id={p} AND canonical_turn_id={p}
                          AND (turn_group_number IS NULL OR turn_group_number<0)""",
                                 (group, conversation_id, row["canonical_turn_id"]))
    except Exception:
        # Matches the old loader's lazy-repair fallback. The read below still
        # derives exact legacy groups; the transaction rolled back any repair.
        logger.warning("Lazy canonical group repair failed for %s", conversation_id[:12], exc_info=True)


def _legacy_groups(rows):
    """Match the merge helper's mixed-legacy content grouping and ordinals."""
    pending = []
    index = 0
    for row in rows:
        user, assistant = row["has_user"], row["has_assistant"]
        if user or not assistant:
            if pending:
                yield index, pending
                index += 1
                pending = []
        if user and not assistant:
            pending = [row]
        elif assistant:
            group = [*pending, row]
            pending = []
            yield index, group
            index += 1
    if pending:
        yield index, pending


def load_uncompacted_groups(store, conversation_id, *, merge_rows, protected_recent_turns=0, limit=None):
    """Hydrate complete pending groups, optionally returning a bounded prefix.

    The protected tail counts only admitted, complete uncompacted logical
    pairs. Full physical siblings are loaded together even when one sibling
    was already compacted. No default truncation changes the public API.
    """
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("Compaction row limit must be a nonnegative integer or None")
    if limit == 0:
        return []
    protected = max(0, int(protected_recent_turns))
    _backfill_all_legacy_groups(store, conversation_id)
    p = store._placeholder
    decoder = store._canonical_decoder()
    result, tail = [], deque()
    with store._relational_connection() as conn:
        smallest, _largest = _group_mode(conn, p, conversation_id)
        legacy = smallest is not None and smallest < 0
        if legacy:
            query = f"""SELECT canonical_turn_id,
                CASE WHEN COALESCE(user_content,'')<>'' THEN 1 ELSE 0 END AS has_user,
                CASE WHEN COALESCE(assistant_content,'')<>'' THEN 1 ELSE 0 END AS has_assistant,
                CASE WHEN compacted_at IS NULL OR compacted_at='' THEN 1 ELSE 0 END AS incomplete
                FROM canonical_turns WHERE conversation_id={p} ORDER BY sort_key,canonical_turn_id"""
        else:
            query = f"""SELECT turn_group_number AS group_id FROM canonical_turns
                WHERE conversation_id={p} AND (compacted_at IS NULL OR compacted_at='')
                GROUP BY turn_group_number ORDER BY turn_group_number"""
        with _scalar_cursor(store, conn, query, (conversation_id,)) as rows:
            if legacy:
                groups = ((group_id, [str(row["canonical_turn_id"]) for row in group])
                          for group_id, group in _legacy_groups(rows)
                          if any(row["incomplete"] for row in group))
            else:
                groups = ((int(row["group_id"]), None) for row in rows)
            while batch := list(islice(groups, 200)):
                if not legacy:
                    values = [group_id for group_id, _ids in batch]
                    predicate = f"turn_group_number IN ({','.join([p] * len(values))})"
                else:
                    group_by_id = {source_id: group_id for group_id, ids in batch for source_id in ids}
                    values = list(group_by_id)
                    predicate = f"canonical_turn_id IN ({','.join([p] * len(values))})"
                raw = conn.execute(f"""SELECT * FROM canonical_turns
                    WHERE conversation_id={p} AND {predicate}
                    ORDER BY sort_key,canonical_turn_id""", [conversation_id, *values]).fetchall()
                physical = [decoder(row) for row in raw]
                if legacy:
                    for row in physical:
                        row.turn_group_number = group_by_id[str(row.canonical_turn_id)]
                for merged in merge_rows(physical).values():
                    if (merged.compacted_at or not (merged.user_content or "").strip()
                            or not (merged.assistant_content or "").strip()):
                        continue
                    tail.append(merged)
                    if len(tail) > protected:
                        result.append(tail.popleft())
                        if limit is not None and len(result) >= limit:
                            return result
    return result


def load_logical_groups(store, conversation_id, turn_numbers, *, merge_rows):
    """Hydrate only requested logical groups, preserving global legacy ordinals."""
    wanted = set(turn_numbers)
    if not wanted:
        return {}
    _backfill_all_legacy_groups(store, conversation_id)
    p = store._placeholder
    decoder = store._canonical_decoder()
    result = {}
    with store._relational_connection() as conn:
        smallest, _largest = _group_mode(conn, p, conversation_id)
        legacy = smallest is not None and smallest < 0
        if not legacy:
            numbers = sorted(wanted)
            for offset in range(0, len(numbers), 200):
                batch = numbers[offset:offset + 200]
                raw = conn.execute(f"""SELECT * FROM canonical_turns
                    WHERE conversation_id={p}
                      AND turn_group_number IN ({','.join([p] * len(batch))})
                    ORDER BY sort_key,canonical_turn_id""", [conversation_id, *batch]).fetchall()
                result.update(merge_rows([decoder(row) for row in raw]))
            return result
        query = f"""SELECT canonical_turn_id,
            CASE WHEN COALESCE(user_content,'')<>'' THEN 1 ELSE 0 END AS has_user,
            CASE WHEN COALESCE(assistant_content,'')<>'' THEN 1 ELSE 0 END AS has_assistant
            FROM canonical_turns WHERE conversation_id={p} ORDER BY sort_key,canonical_turn_id"""
        with _scalar_cursor(store, conn, query, (conversation_id,)) as rows:
            last = max(wanted)
            selected = {}
            def hydrate():
                if not selected:
                    return
                ids = list(selected)
                raw = conn.execute(f"""SELECT * FROM canonical_turns
                    WHERE conversation_id={p}
                      AND canonical_turn_id IN ({','.join([p] * len(ids))})
                    ORDER BY sort_key,canonical_turn_id""", [conversation_id, *ids]).fetchall()
                physical = [decoder(row) for row in raw]
                for row in physical:
                    row.turn_group_number = selected[str(row.canonical_turn_id)]
                result.update(merge_rows(physical))
                selected.clear()
            for group_id, group in _legacy_groups(rows):
                if group_id > last:
                    break
                if group_id in wanted:
                    selected.update({str(row["canonical_turn_id"]): group_id for row in group})
                    if len(selected) >= 200:
                        hydrate()
            hydrate()
    return result
