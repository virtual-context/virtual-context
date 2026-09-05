"""Real SQLite contracts shared by paged reads and durable continuations."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnChunkEmbedding, ChunkEmbedding, StoredSegment


@pytest.fixture
def store(tmp_path):
    result = SQLiteStore(tmp_path / "contracts.db")
    yield result
    result.close()


def test_embedding_pages_scope_before_decode_and_continue_in_key_order(store):
    for ref, owner in [('a','mine'),('b','mine'),('z','foreign')]:
        store.store_segment(StoredSegment(ref=ref, conversation_id=owner))
        store.store_chunk_embeddings(ref,[ChunkEmbedding(segment_ref=ref,chunk_index=n,text=f'{ref}-{n}',embedding=[1.0,n]) for n in range(3)])
    store._get_conn().execute("UPDATE segment_chunks SET embedding_json='corrupt' WHERE segment_ref='z'")
    result, cursor = [], None
    while page := store.get_segment_chunk_embedding_page(conversation_id='mine',limit=2,after=cursor):
        assert len(page) <= 2
        result.extend(item['text'] for item in page)
        cursor = page[-1]['cursor']
    assert result == ['a-0','a-1','a-2','b-0','b-1','b-2']


def test_canonical_pages_hydrate_exact_physical_sources(store):
    for turn in range(3):
        key=f'physical-{turn}'
        store.save_canonical_turn('mine',turn,f'user-{turn}',f'assistant-{turn}',canonical_turn_id=key,turn_group_number=turn,source_message_id='same-source' if turn<2 else 'other')
        chunk=CanonicalTurnChunkEmbedding(conversation_id='mine',canonical_turn_id=key,turn_number=turn,side='user',chunk_index=0,text=f'user-{turn}',embedding=[1.0])
        store.store_canonical_turn_chunk_embeddings('mine',turn,'user',[chunk],canonical_turn_id=key)
    cursor, ids = None, []
    while page := store.get_canonical_turn_chunk_embedding_page(conversation_id='mine',limit=1,after=cursor):
        row=page[0]
        assert row['physical_row'].user_content == row['text']
        assert row['physical_row'].canonical_turn_id == row['canonical_turn_id']
        ids.append(row['canonical_turn_id']); cursor=row['cursor']
    assert ids == ['physical-0','physical-1','physical-2']
    assert len(store.get_canonical_turn_rows_by_source_message_ids('mine',['same-source'],internal_validation=True)) == 2
    assert [row.canonical_turn_id for row in store.get_canonical_turn_rows_by_group('mine',[2],internal_validation=True)] == ['physical-2']
    with pytest.raises(PermissionError):
        store.get_canonical_turn_rows_by_group('mine',[0])


@pytest.mark.parametrize('legacy', [False, True])
def test_compaction_watermark_stops_at_first_incomplete_pair(store, legacy):
    for turn in range(3):
        store.save_canonical_turn('c',turn,f'user-{turn}',f'assistant-{turn}',canonical_turn_id=f'p{turn}',turn_group_number=turn)
    conn=store._get_conn()
    conn.execute("UPDATE canonical_turns SET compacted_at='2026-09-05' WHERE canonical_turn_id IN ('p0','p2')")
    if legacy:
        conn.execute("UPDATE canonical_turns SET turn_group_number=-1")
    assert store.get_compaction_watermark('c') == (2,0)
    assert store.get_compaction_watermark('empty') == (0,-1)
    conn.execute("UPDATE canonical_turns SET user_content=? WHERE canonical_turn_id='p0'",('\u00a0\u2003\t',))
    assert store.get_compaction_watermark('c') == (0,-1)


def test_pending_exchange_survives_worker_change_and_has_exclusive_lease(store):
    other=SQLiteStore(store.db_path)
    now=time.time()
    payload=json.dumps({'authority':'owner','hidden_result':'source evidence'})
    try:
        assert store.put_pending_exchange('owner','exchange',payload,expires_at=now+500)
        assert other.list_pending_exchanges('owner',now=now) == ['exchange']
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(s.claim_pending_exchange,'owner','exchange',name,now=now,lease_seconds=5) for s,name in [(store,'a'),(other,'b')]]
            results=[future.result() for future in futures]
        assert results.count(payload) == 1 and results.count(None) == 1
        winner='a' if results[0] else 'b'
        assert not other.finish_pending_exchange('owner','exchange','wrong',consume=True)
        assert other.renew_pending_exchange('owner','exchange',winner,now=now+1,lease_seconds=120)
        assert store.claim_pending_exchange('owner','exchange','new',now=now+6) is None
        assert other.finish_pending_exchange('owner','exchange',winner,consume=False)
        assert store.claim_pending_exchange('owner','exchange','new',now=now+7) == payload
        assert store.finish_pending_exchange('owner','exchange','new',consume=True)
        assert other.list_pending_exchanges('owner',now=now+8) == []
    finally:
        other.close()


def test_pending_exchange_capacity_expiry_and_conversation_deletion(store):
    now=time.time()
    for number in range(4):
        assert store.put_pending_exchange('owner',str(number),'{}',expires_at=now+60)
    assert not store.put_pending_exchange('owner','overflow','{}',expires_at=now+60)
    assert not store.put_pending_exchange('other','too-large',json.dumps('x'*20),expires_at=now+60,max_bytes=10)
    assert store.claim_pending_exchange('owner','0','claim',now=now+61) is None
    assert store.claim_pending_exchange('foreign','0','claim',now=now) is None
    store.delete_conversation('owner')
    assert store.list_pending_exchanges('owner',now=now) == []
