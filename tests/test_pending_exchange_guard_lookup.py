"""Looking up nonexistent continuation IDs must never create lifecycle rows."""

import time

import pytest

from virtual_context.storage.sqlite import SQLiteStore


@pytest.mark.parametrize('known_conversation', [False, True])
def test_unknown_checkpoint_guards_leave_lifecycle_tables_unchanged(tmp_path, known_conversation):
    store = SQLiteStore(tmp_path / 'exchanges.db')
    try:
        if known_conversation:
            store.upsert_conversation(tenant_id='tenant', conversation_id='owner')
        conn = store._get_conn()
        before = [tuple(row) for row in conn.execute('SELECT * FROM conversation_lifecycle')]
        now = time.time()
        assert store.get_pending_exchange('owner', 'missing', now=now) is None
        assert store.list_pending_exchanges('owner', now=now) == []
        assert store.claim_pending_exchange('owner', 'missing', 'claim', now=now) is None
        assert store.renew_pending_exchange('owner', 'missing', 'claim', now=now) is False
        assert store.finish_pending_exchange('owner', 'missing', 'claim', consume=True) is False
        assert store.finish_pending_exchange('owner', 'missing', 'claim', consume=False) is False
        assert [tuple(row) for row in conn.execute('SELECT * FROM conversation_lifecycle')] == before
        assert conn.execute('SELECT count(*) FROM pending_tool_exchanges').fetchone()[0] == 0
    finally:
        store.close()


def test_first_checkpoint_still_materializes_and_fences_a_local_owner(tmp_path):
    store = SQLiteStore(tmp_path / 'exchanges.db')
    try:
        now = time.time()
        assert store.put_pending_exchange('local-owner', 'exchange', '{}', expires_at=now + 600)
        conn = store._get_conn()
        assert conn.execute('SELECT count(*) FROM conversation_lifecycle WHERE conversation_id=?', ('local-owner',)).fetchone()[0] == 1
        assert store.claim_pending_exchange('local-owner', 'exchange', 'claim', now=now) == '{}'
        conn.execute('UPDATE conversation_lifecycle SET generation=generation+1 WHERE conversation_id=?', ('local-owner',))
        assert store.renew_pending_exchange('local-owner', 'exchange', 'claim', now=now) is False
        assert store.finish_pending_exchange('local-owner', 'exchange', 'claim', consume=True) is False
        assert store.get_pending_exchange('local-owner', 'exchange', now=now) is None
    finally:
        store.close()
