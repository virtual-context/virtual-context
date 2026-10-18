"""Postgres twin of compacted raw-body storage (BUG-131)."""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set",
)


def _image(seed: str) -> str:
    return "data:image/png;base64," + base64.b64encode((seed * 4000).encode()).decode()


def _body(words: str, images: list[str]) -> str:
    return json.dumps(
        [{"type": "input_text", "text": words}]
        + [{"type": "input_image", "image_url": image} for image in images]
    )


@pytest.mark.regression("BUG-131")
def test_postgres_saves_backfills_and_collects_raw_blobs(monkeypatch):
    from virtual_context.core.raw_bodies import compact_raw_body, expand_raw_body
    from virtual_context.storage import raw_blobs
    from virtual_context.storage.postgres import PostgresStore

    monkeypatch.setattr(raw_blobs, "GC_GRACE", timedelta(0))
    suffix = uuid.uuid4().hex
    conv = f"sk:agent:test:discord:guild:{suffix}"
    store = PostgresStore(PG_URL)
    store.activate_conversation(conv)
    user_id, assistant_id, plain_id = (str(uuid.uuid4()) for _ in range(3))
    mine, shared = _image(f"mine-{suffix}"), _image(f"shared-{suffix}")
    attested_raw = _body("attested words", [mine, shared])
    provenance = dict(
        sender_actor_id="actor", source_message_id=f"m-{suffix}",
        audience_conversation_id=conv, origin_channel_id="ch",
    )
    try:
        # Rows written before compaction existed: store the raw body directly.
        store.save_canonical_turn(
            conv, -1, "user words", "", user_raw_content="x",
            canonical_turn_id=user_id, sort_key=1000.0, turn_hash=f"hu-{suffix}",
            turn_group_number=0, **provenance,
        )
        store.save_canonical_turn(
            conv, -1, "", "reply", canonical_turn_id=assistant_id, sort_key=1001.0,
            turn_hash=f"ha-{suffix}", turn_group_number=0,
        )
        store.save_canonical_turn(
            conv, -1, "plain", "", user_raw_content="x", canonical_turn_id=plain_id,
            sort_key=2000.0, turn_hash=f"hp-{suffix}", turn_group_number=1,
        )
        now = datetime.now(timezone.utc).isoformat()
        with store.pool.connection() as conn:
            conn.execute(
                "UPDATE canonical_turns SET user_raw_content = %s WHERE canonical_turn_id = ANY(%s)",
                (attested_raw, [user_id, plain_id]),
            )
            conn.execute(
                """INSERT INTO canonical_message_sources (tenant_id, agent_scope_id, platform,
                   account_id, message_id, canonical_turn_id, assistant_canonical_turn_id,
                   assistant_turn_hash, pair_version, audience_conversation_id, channel_id,
                   guild_id, author_id, source_actor_id, transport_body_sha256,
                   canonical_body_sha256, projection_version, canonical_turn_hash,
                   observed_at, created_at)
                   VALUES ('t', 'a', 'discord', 'acc', %s, %s, %s, %s, 1, %s, 'ch', 'g', 'au',
                           'actor', 'x', %s, '1', %s, %s, %s)""",
                (f"m-{suffix}", user_id, assistant_id, f"ha-{suffix}", conv,
                 hashlib.sha256(b"user words").hexdigest(), f"hu-{suffix}", now, now),
            )

        counts = store.compact_stored_raw_bodies(conv, batch_size=1)
        assert counts["rows_compacted"] == 2
        compacted = compact_raw_body(attested_raw).text
        with store.pool.connection() as conn:
            stored = {
                str(row["canonical_turn_id"]): row["user_raw_content"]
                for row in conn.execute(
                    "SELECT canonical_turn_id, user_raw_content FROM canonical_turns WHERE conversation_id = %s",
                    (conv,),
                ).fetchall()
            }
            grants = conn.execute("SELECT COUNT(*) AS n FROM canonical_raw_compaction_grants").fetchone()["n"]
        assert stored[user_id] == compacted and stored[plain_id] == compacted
        assert grants == 0
        expanded = json.loads(expand_raw_body(compacted, store.get_raw_blobs))
        assert [part["image_url"] for part in expanded[1:]] == [mine, shared]

        # Another conversation shares one image; deleting this one keeps it.
        other = f"{conv}-other"
        store.activate_conversation(other)
        store.save_canonical_turn(
            other, -1, "other", "", user_raw_content=_body("other", [shared]),
            canonical_turn_id=str(uuid.uuid4()), sort_key=1000.0,
            turn_hash=f"ho-{suffix}", turn_group_number=0,
        )
        store.delete_conversation(conv)
        left = set(store.get_raw_blobs({
            hashlib.sha256(mine.encode()).hexdigest(),
            hashlib.sha256(shared.encode()).hexdigest(),
        }))
        assert left == {hashlib.sha256(shared.encode()).hexdigest()}
        store.delete_conversation(other)
    finally:
        store.delete_conversation(conv)
