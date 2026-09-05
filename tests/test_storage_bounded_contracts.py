"""Execute bounded SQL reads and proposal round-trips on both real backends."""
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid

import pytest

from tests.test_storage_domain_contracts import store as store
from virtual_context.core.fact_lifecycle import fact_version
from virtual_context.ingest.supersession import _proposal_snapshot
from virtual_context.types import CanonicalTurnChunkEmbedding, ChunkEmbedding, Fact, StoredSegment


def _pair(store, group):
    keys = []
    for index, (user, assistant) in enumerate([(f'user {group}', ''), ('', f'assistant {group}')]):
        key = str(uuid.uuid4())
        store.save_canonical_turn('owner', group*2+index, user, assistant,
                                 canonical_turn_id=key, turn_group_number=group,
                                 sort_key=float(group*2+index), source_message_id=f'message-{group}')
        keys.append(key)
    return keys


def _execute(store, query, params=()):
    with store._relational_connection(write=True, scope='bounded-test-seed') as conn:
        conn.execute(query.replace('?', store._placeholder), params)


def _spy_named_cursors(store, monkeypatch):
    events = []
    if store._relational_dialect != 'postgres':
        return events
    original = store._relational_connection
    class Cursor:
        def __init__(self, cursor, name):
            self.raw, self.name = cursor, name
        def __getattr__(self, key):
            return getattr(self.raw, key)
        def fetchmany(self, size):
            rows = self.raw.fetchmany(size)
            events.append((self.name, size, len(rows)))
            assert size == 200
            assert all('user_content' not in row and 'assistant_content' not in row for row in rows)
            return rows
    class Connection:
        def __init__(self, connection):
            self.raw = connection
        def __getattr__(self, key):
            return getattr(self.raw, key)
        def cursor(self, *args, **kwargs):
            cursor = self.raw.cursor(*args, **kwargs)
            return Cursor(cursor, kwargs['name']) if kwargs.get('name') else cursor
    @contextmanager
    def connection(**kwargs):
        with original(**kwargs) as conn:
            yield Connection(conn)
    monkeypatch.setattr(store, '_relational_connection', connection)
    return events


@pytest.mark.parametrize('legacy', [False, True])
def test_scalar_watermark_named_cursor_and_unicode_trim(store, monkeypatch, legacy):
    for group in range(3):
        _pair(store, group)
    _execute(store, "UPDATE canonical_turns SET compacted_at=? WHERE turn_group_number IN (0,2)", ('2026-09-05T00:00:00+00:00',))
    if legacy:
        _execute(store, 'UPDATE canonical_turns SET turn_group_number=-1')
    events = _spy_named_cursors(store, monkeypatch)
    assert store.get_compaction_watermark('owner') == (2, 0)
    assert store.get_compaction_watermark('foreign') == (0, -1)
    _execute(store, 'UPDATE canonical_turns SET user_content=? WHERE sort_key=0', ('\u00a0\u2003\t',))
    assert store.get_compaction_watermark('owner') == (0, -1)
    if store._relational_dialect == 'postgres':
        assert events and all(name.startswith('vc_watermark_') for name, _, _ in events)


def test_all_legacy_backfill_interleaves_server_cursor_and_writes(store, monkeypatch):
    keys = [_pair(store, group) for group in range(3)]
    _execute(store, 'UPDATE canonical_turns SET turn_group_number=-1')
    monkeypatch.setattr(store, '_load_canonical_turn_rows', lambda *_: pytest.fail('full archive hydration'))
    monkeypatch.setattr(store, 'recompute_canonical_turn_groups', lambda *_: pytest.fail('unbounded legacy repair'))
    events = _spy_named_cursors(store, monkeypatch)
    selected = store.get_uncompacted_canonical_turns('owner', limit=1)
    assert len(selected) == 1 and selected[0].turn_group_number == 0
    with store._relational_connection() as conn:
        rows = conn.execute('SELECT canonical_turn_id,turn_group_number FROM canonical_turns ORDER BY sort_key').fetchall()
    assert [(str(row['canonical_turn_id']), row['turn_group_number']) for row in rows] == [
        (key, group) for group, pair in enumerate(keys) for key in pair
    ]
    assert store.mark_canonical_turns_compacted('owner', [selected[0].canonical_turn_id]) == 2
    assert [row.turn_group_number for row in store.get_uncompacted_canonical_turns('owner')] == [1, 2]
    if store._relational_dialect == 'postgres':
        assert events and all(name.startswith('vc_pending_') for name, _, _ in events)


def test_pending_hydration_excludes_archive_and_retains_compacted_sibling(store, monkeypatch):
    keys = [_pair(store, group) for group in range(5)]
    _execute(store, 'UPDATE canonical_turns SET compacted_at=? WHERE turn_group_number<3 OR canonical_turn_id=?',
             ('2026-09-05T00:00:00+00:00', keys[3][0]))
    decoder, decoded = store._canonical_decoder(), []
    def observed(row):
        decoded.append(str(row['canonical_turn_id']))
        assert row['turn_group_number'] >= 3
        return decoder(row)
    monkeypatch.setattr(store, '_canonical_decoder', lambda: observed)
    selected = store.get_uncompacted_canonical_turns('owner', protected_recent_turns=1, limit=1)
    assert [row.turn_group_number for row in selected] == [3]
    assert selected[0].user_content == 'user 3' and selected[0].assistant_content == 'assistant 3'
    assert set(decoded) == set(keys[3] + keys[4])


def test_embedding_pages_and_physical_source_reads_keep_owner_scope(store):
    for ref, owner in [('a', 'owner'), ('b', 'owner'), ('z', 'foreign')]:
        store.store_segment(StoredSegment(ref=ref, conversation_id=owner))
        store.store_chunk_embeddings(ref, [ChunkEmbedding(segment_ref=ref, chunk_index=i, text=f'{ref}-{i}', embedding=[1.0, float(i)]) for i in range(3)])
    _execute(store, "UPDATE segment_chunks SET embedding_json='broken-json' WHERE segment_ref='z'")
    cursor, texts = None, []
    while page := store.get_segment_chunk_embedding_page(conversation_id='owner', limit=2, after=cursor):
        assert len(page) <= 2
        texts.extend(row['text'] for row in page)
        cursor = page[-1]['cursor']
    assert texts == ['a-0','a-1','a-2','b-0','b-1','b-2']
    keys = _pair(store, 0)
    for number, key in enumerate(keys):
        store.store_canonical_turn_chunk_embeddings('owner', number, 'user', [CanonicalTurnChunkEmbedding(
            conversation_id='owner', canonical_turn_id=key, turn_number=number, side='user',
            chunk_index=0, text=f'chunk-{number}', embedding=[1.0])], canonical_turn_id=key)
    cursor, actual = None, []
    while page := store.get_canonical_turn_chunk_embedding_page(conversation_id='owner', limit=1, after=cursor):
        actual.append(page[0]['canonical_turn_id'])
        assert page[0]['physical_row'].canonical_turn_id == page[0]['canonical_turn_id']
        cursor = page[-1]['cursor']
    assert actual == keys
    assert {row.canonical_turn_id for row in store.get_canonical_turn_rows_by_source_message_ids('owner', ['message-0'], internal_validation=True)} == set(keys)
    assert {row.canonical_turn_id for row in store.get_canonical_turn_rows_by_group('owner', [0], internal_validation=True)} == set(keys)
    assert store.get_canonical_turn_rows_by_source_message_ids('foreign', ['message-0'], internal_validation=True) == []
    with pytest.raises(PermissionError):
        store.get_canonical_turn_rows_by_group('owner', [0])


def test_new_fact_proposal_snapshot_survives_database_timestamp_roundtrip(store):
    fact = Fact(id='roundtrip', conversation_id='owner', subject='Alice', verb='likes', object='tea',
                mentioned_at=datetime(2026, 9, 5, 12, 34, 56, 123456, tzinfo=timezone.utc))
    store.store_facts([fact])
    snapshot = _proposal_snapshot(store, fact)
    assert snapshot is not None
    assert snapshot['fact_version'] == fact_version(fact)
    assert snapshot == store.get_fact_admission_snapshot(fact.id)
