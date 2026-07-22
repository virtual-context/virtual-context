"""Production-dialect coverage for guarded canonical actor-id repair."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.pg_helpers import pg_dsn


PG_URL = pg_dsn()
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_postgres_actor_normalization_dry_run_rollback_apply_and_replay():
    from psycopg import sql
    from virtual_context.storage.postgres import PostgresStore

    suffix = uuid.uuid4().hex
    tenant = f"tenant-actor-normalize-{suffix}"
    owner = f"sk:agent:test:discord:guild:{suffix}"
    trigger_name = f"actor_norm_fail_{suffix}"
    function_name = f"actor_norm_fail_fn_{suffix}"
    sender_turn_id = str(uuid.uuid4())
    reply_turn_id = str(uuid.uuid4())
    now = _now()
    store = PostgresStore(PG_URL)
    store.activate_conversation(owner)
    store.upsert_conversation(tenant_id=tenant, conversation_id=owner)
    assert store.set_phase(
        conversation_id=owner, lifecycle_epoch=1, phase="active",
    )

    store.save_canonical_turn(
        owner, -1, "legacy sender", "",
        canonical_turn_id=sender_turn_id, sort_key=1000.0,
        turn_hash=f"sender-hash-{suffix}", sender_actor_id=" 42 ",
        primary_tag="chat", tags=["chat"], tagged_at=now,
        audience_conversation_id=owner, audience_attribution_version=1,
        origin_channel_id="111",
    )
    store.save_canonical_turn(
        owner, -1, "legacy reply", "",
        canonical_turn_id=reply_turn_id, sort_key=2000.0,
        turn_hash=f"reply-hash-{suffix}",
        sender_actor_id="actor:discord:99", reply_subject_actor_id="42",
        primary_tag="chat", tags=["chat"], tagged_at=now,
        audience_conversation_id=owner, audience_attribution_version=1,
        origin_channel_id="111",
    )
    assert store.upsert_actor_profile_from_turn(
        owner, "actor:discord:42", "Legacy", seen_at=now,
        expected_lifecycle_epoch=1,
    )

    try:
        preview = store.normalize_canonical_actor_ids(
            owner, tenant_id=tenant, expected_lifecycle_epoch=1,
            platform="discord",
        )
        assert preview["sender_rows_to_normalize"] == 1
        assert preview["reply_subject_rows_to_normalize"] == 1
        assert preview["selected_rows"] == 2
        assert preview["distinct_actor_ids"] == 1
        assert preview["updated_rows"] == 0
        assert not preview["platform_mismatch_sources"]

        # A matching bare profile makes the identity merge ambiguous.
        with store.pool.connection() as conn:
            conn.execute(
                """INSERT INTO actor_profiles
                   (tenant_id, actor_id, platform, display_name,
                    first_seen_at, last_seen_at)
                   VALUES (%s, '42', 'discord', 'Bare', %s, %s)""",
                (tenant, now, now),
            )
        with pytest.raises(RuntimeError, match="explicit profile merge"):
            store.normalize_canonical_actor_ids(
                owner, tenant_id=tenant, expected_lifecycle_epoch=1,
                platform="discord", dry_run=False,
            )
        with store.pool.connection() as conn:
            conn.execute(
                "DELETE FROM actor_profiles WHERE tenant_id = %s AND actor_id = '42'",
                (tenant,),
            )

        # A parsed alias from another platform blocks operator-wide prefixing.
        mixed_alias = f"sk:agent:test:telegram:group:{suffix}"
        store.save_conversation_alias(mixed_alias, owner)
        mixed = store.normalize_canonical_actor_ids(
            owner, tenant_id=tenant, expected_lifecycle_epoch=1,
            platform="discord",
        )
        assert mixed["platform_mismatch_sources"] == [mixed_alias]
        with pytest.raises(RuntimeError, match="another platform"):
            store.normalize_canonical_actor_ids(
                owner, tenant_id=tenant, expected_lifecycle_epoch=1,
                platform="discord", dry_run=False,
            )
        store.delete_conversation_alias(mixed_alias)

        # A failure after the canonical UPDATE must roll the entire transaction
        # back.  The tenant-filtered trigger cannot affect unrelated tests.
        with store.pool.connection() as conn:
            conn.execute(sql.SQL("""
                CREATE FUNCTION {}() RETURNS trigger AS $$
                BEGIN
                    IF NEW.tenant_id = {} THEN
                        RAISE EXCEPTION 'forced actor normalization rollback';
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """).format(
                sql.Identifier(function_name), sql.Literal(tenant),
            ))
            conn.execute(sql.SQL("""
                CREATE TRIGGER {} BEFORE UPDATE ON actor_profiles
                FOR EACH ROW EXECUTE FUNCTION {}()
            """).format(
                sql.Identifier(trigger_name), sql.Identifier(function_name),
            ))
        with pytest.raises(Exception, match="forced actor normalization rollback"):
            store.normalize_canonical_actor_ids(
                owner, tenant_id=tenant, expected_lifecycle_epoch=1,
                platform="discord", dry_run=False,
            )
        with store.pool.connection() as conn:
            rows = conn.execute(
                """SELECT sender_actor_id, reply_subject_actor_id
                     FROM canonical_turns WHERE conversation_id = %s
                     ORDER BY sort_key""",
                (owner,),
            ).fetchall()
        assert rows[0]["sender_actor_id"].strip() == "42"
        assert rows[1]["reply_subject_actor_id"] == "42"

        with store.pool.connection() as conn:
            conn.execute(sql.SQL("DROP TRIGGER {} ON actor_profiles").format(
                sql.Identifier(trigger_name),
            ))
            conn.execute(sql.SQL("DROP FUNCTION {}()").format(
                sql.Identifier(function_name),
            ))

        applied = store.normalize_canonical_actor_ids(
            owner, tenant_id=tenant, expected_lifecycle_epoch=1,
            platform="discord", dry_run=False,
        )
        assert applied["updated_rows"] == 2
        replay = store.normalize_canonical_actor_ids(
            owner, tenant_id=tenant, expected_lifecycle_epoch=1,
            platform="discord", dry_run=False,
        )
        assert replay["selected_rows"] == replay["updated_rows"] == 0
        with store.pool.connection() as conn:
            rows = conn.execute(
                """SELECT sender_actor_id, reply_subject_actor_id
                     FROM canonical_turns WHERE conversation_id = %s
                     ORDER BY sort_key""",
                (owner,),
            ).fetchall()
        assert rows[0]["sender_actor_id"] == "actor:discord:42"
        assert rows[1]["reply_subject_actor_id"] == "actor:discord:42"
        assert rows[1]["sender_actor_id"] == "actor:discord:99"
    finally:
        # Cleanup is best-effort because the assertion path may leave the test
        # trigger installed.  Names and tenant are unique to this test.
        with store.pool.connection() as conn:
            conn.execute(sql.SQL("DROP TRIGGER IF EXISTS {} ON actor_profiles").format(
                sql.Identifier(trigger_name),
            ))
            conn.execute(sql.SQL("DROP FUNCTION IF EXISTS {}()").format(
                sql.Identifier(function_name),
            ))
        store.delete_conversation(owner)
        with store.pool.connection() as conn:
            conn.execute(
                "DELETE FROM actor_profiles WHERE tenant_id = %s", (tenant,),
            )
        store.close()
