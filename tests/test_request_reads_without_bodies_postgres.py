"""Postgres twin of the body-free request reads (BUG-130).

Postgres returns ``canonical_turn_id`` as a UUID, so a projection that keys
rows by the raw column value never matches the string keys the caller asks
for, and every summary would pass the audience check unchecked.
"""
from __future__ import annotations

import json
import uuid

import pytest

from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set",
)


@pytest.mark.regression("BUG-130")
def test_postgres_projections_match_full_rows():
    from virtual_context.storage.postgres import PostgresStore

    suffix = uuid.uuid4().hex
    conv = f"sk:agent:test:discord:guild:{suffix}"
    store = PostgresStore(PG_URL)
    store.activate_conversation(conv)
    ids = [str(uuid.uuid4()) for _ in range(3)]
    raw = json.dumps([{"type": "input_image", "image_url": "data:image/png;base64," + "QUJD" * 500}])
    try:
        for i, turn_id in enumerate(ids):
            store.save_canonical_turn(
                conv, -1, f"user {i}", f"assistant {i}",
                user_raw_content=raw,
                canonical_turn_id=turn_id, sort_key=1000.0 * (i + 1),
                turn_hash=f"hash-{suffix}-{i}", turn_group_number=i,
                audience_conversation_id="other-guild" if i == 0 else conv,
                audience_attribution_version=1,
            )
        keys = [(conv, turn_id) for turn_id in ids]
        full = store.get_canonical_turn_rows_by_id(keys, internal_validation=True)
        assert store.get_canonical_turn_audiences(keys) == {
            key: (row.audience_conversation_id, row.audience_attribution_version)
            for key, row in full.items()
        }
        recent = store.get_recent_canonical_turns(conv, limit=3)
        assert recent and all(row.user_raw_content is None for row in recent)
        texts = store.get_all_canonical_turn_texts(conv)
        assert [row.user_content for row in texts] == ["user 0", "user 1", "user 2"]
        assert all(row.user_raw_content is None for row in texts)
    finally:
        store.delete_conversation(conv)


@pytest.mark.regression("BUG-143")
def test_postgres_index_projection_hashes_match_python():
    import hashlib

    from virtual_context.engine import VirtualContextEngine
    from virtual_context.storage.postgres import PostgresStore

    suffix = uuid.uuid4().hex
    conv = f"sk:agent:test:discord:guild:{suffix}"
    store = PostgresStore(PG_URL)
    store.activate_conversation(conv)
    try:
        turns = [
            (0, "naïve café — 東京 ✓", "réponse ü"),
            (1, "", ""),
            (1, "second user", ""),
            (1, "", "second answer"),
            (2, "  ", "only blank user"),
        ]
        for i, (group, user, assistant) in enumerate(turns):
            store.save_canonical_turn(
                conv, -1, user, assistant,
                canonical_turn_id=str(uuid.uuid4()), sort_key=1000.0 * (i + 1),
                turn_hash=f"hash-{suffix}-{i}", turn_group_number=group,
            )
        rows = store.get_canonical_turn_index_rows(conv)
        full = store.get_all_canonical_turns(conv)
        assert [r.canonical_turn_id for r in rows] == [r.canonical_turn_id for r in full]
        expected = {}
        for group, pair_rows in VirtualContextEngine._group_canonical_rows_into_pairs(full):
            user, assistant, *_ = VirtualContextEngine._pair_payload_from_rows(pair_rows)
            expected[group] = hashlib.sha256(f"{user} {assistant}".encode()).hexdigest()[:16]
        assert {r.turn_group_number: r.pair_hash for r in rows} == expected
        assert [(r.has_user_content, r.has_assistant_content) for r in rows] == [
            (bool(r.user_content.strip()), bool(r.assistant_content.strip())) for r in full
        ]
    finally:
        store.delete_conversation(conv)
