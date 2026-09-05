"""Shared relational contracts, run against SQLite and isolated fleet databases.

No server is started here. Set VC_REQUIRE_STORAGE_DOMAIN_TESTS=1 for a mandatory
remote gate: missing DSN/CREATE DATABASE permission then fails rather than skips.
A skipped PostgreSQL cell is not evidence of PostgreSQL conformance.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import os
import sqlite3
import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.core.exceptions import ConversationLifecycleConflict
from virtual_context.core.store_capabilities import (
    RELATIONAL_CAPABILITIES, StoreCapabilities, capabilities_of,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CompactionLeaseLost, Fact,
    SegmentMetadata, StoredSegment,
)


@pytest.fixture(params=['sqlite', 'postgres'])
def store(request, tmp_path):
    if request.param == 'sqlite':
        result = SQLiteStore(tmp_path / 'domains.db', compaction_fence_mode=CompactionFenceMode.ACTIVE)
        try:
            yield result
        finally:
            result.close()
        return

    required = os.environ.get('VC_REQUIRE_STORAGE_DOMAIN_TESTS') == '1'
    if not pg_dsn():
        if required:
            pytest.fail('Required storage-domain fleet run has no configured DSN')
        pytest.skip('PostgreSQL fleet DSN not configured')
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from virtual_context.storage.postgres import PostgresStore

    admin = pg_test_conn()
    name = f'vc_domain_test_{uuid.uuid4().hex}'
    try:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0 ENCODING 'UTF8'").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        if required:
            pytest.fail('Required storage-domain fleet role cannot create isolated test databases')
        pytest.skip('fleet role cannot create isolated storage-domain test database')
    result = None
    try:
        params = conninfo_to_dict(pg_dsn())
        params['dbname'] = name
        result = PostgresStore(make_conninfo(**params), compaction_fence_mode=CompactionFenceMode.ACTIVE)
        yield result
    finally:
        if result is not None:
            result.close()
        admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def _fact(store, key='old', *, date='2026-01-01', **kwargs):
    fact = Fact(id=key, conversation_id='owner', subject='Alice', verb='lives in',
                object='Boston' if key == 'old' else 'Denver', when_date=date, **kwargs)
    store.store_facts([fact])
    return fact


def _physical_fact(store, key, *, ordinal=0, date='2026-01-01'):
    canonical_id = str(uuid.uuid4())
    what = f'Alice lives in {"Boston" if key == "old" else "Denver"}'
    store.save_canonical_turn(
        'owner', ordinal, what, '', canonical_turn_id=canonical_id,
        sort_key=float(ordinal), sender_actor_id='alice', source_message_id=f'message-{key}',
        audience_conversation_id='audience', audience_attribution_version=1,
        origin_channel_id='channel',
    )
    store.store_segment(StoredSegment(
        ref=f'segment-{key}', conversation_id='owner', full_text=what, summary=what,
        metadata=SegmentMetadata(canonical_turn_ids=[canonical_id], source_mapping_complete=True),
    ))
    fact = _fact(store, key, date=date, what=what, segment_ref=f'segment-{key}',
                 author_actor_id='alice', author_attribution_version=2,
                 author_source_role='requester', author_source_message_id=f'message-{key}')
    return fact, canonical_id


def _saved(store, key):
    with store._relational_connection() as conn:
        return dict(conn.execute(f'SELECT * FROM facts WHERE id={store._placeholder}', (key,)).fetchone())


@contextmanager
def _reject_audit_insert(store):
    """Fail the actual last SQL write, after the projection/cache have changed."""
    with store._relational_connection(write=True, scope='install-test-trigger') as conn:
        if store._relational_dialect == 'postgres':
            conn.execute("""CREATE FUNCTION reject_domain_audit() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'injected audit insert failure'; END; $$""")
            conn.execute('CREATE TRIGGER reject_domain_audit BEFORE INSERT ON fact_decisions FOR EACH ROW EXECUTE FUNCTION reject_domain_audit()')
        else:
            conn.execute("""CREATE TRIGGER reject_domain_audit BEFORE INSERT ON fact_decisions
                BEGIN SELECT RAISE(ABORT, 'injected audit insert failure'); END""")
    try:
        yield
    finally:
        with store._relational_connection(write=True, scope='remove-test-trigger') as conn:
            if store._relational_dialect == 'postgres':
                conn.execute('DROP TRIGGER reject_domain_audit ON fact_decisions')
                conn.execute('DROP FUNCTION reject_domain_audit()')
            else:
                conn.execute('DROP TRIGGER reject_domain_audit')


def test_capabilities_describe_backend_and_require_explicit_guarantees(store):
    capabilities = capabilities_of(store)
    expected = asdict(RELATIONAL_CAPABILITIES)
    expected['native_vectors'] = store._relational_dialect == 'postgres'
    assert asdict(capabilities) == expected
    capabilities.require(*(name for name, enabled in expected.items() if enabled))
    with pytest.raises(ValueError, match='Unknown storage capabilities: imaginary'):
        capabilities.require('imaginary')
    if not capabilities.native_vectors:
        with pytest.raises(ValueError, match='Storage does not provide: native_vectors'):
            capabilities.require('native_vectors')
    else:
        # Advertising implementation capability must not activate unmigrated data.
        assert not store.vector_search_ready('all-MiniLM-L6-v2')
    assert capabilities_of(object()) == StoreCapabilities()
    with pytest.raises(ValueError, match='Storage does not provide: conversation_scope'):
        capabilities_of(object()).require('conversation_scope')


@pytest.mark.parametrize('incoming_date,accepted,reason', [
    ('2027-01-01', True, 'legacy_unattributed'),
    ('2025-01-01', False, 'older_evidence'),
])
def test_supersession_records_accepted_and_rejected_proposals(store, incoming_date, accepted, reason):
    old = _fact(store)
    new = _fact(store, 'new', date=incoming_date)
    assert store.set_fact_superseded(old.id, new.id) is accepted
    row = _saved(store, old.id)
    assert row['superseded_by'] == (new.id if accepted else None)
    audit, = store.get_fact_decisions('owner')
    assert (audit['accepted'], audit['reason'], audit['action']) == (int(accepted), reason, 'supersede')
    assert audit['event_date'] == incoming_date
    assert json.loads(audit['before_json'])['superseded_by'] is None
    assert json.loads(audit['after_json'])['superseded_by'] == row['superseded_by']
    assert json.loads(audit['source_versions_json']) == []
    assert store.get_fact_decisions('foreign') == []


@pytest.mark.parametrize('corrupt_source', [False, True])
def test_each_attributed_endpoint_requires_its_own_current_author_proof(store, corrupt_source):
    old, old_id = _physical_fact(store, 'old')
    new, new_id = _physical_fact(store, 'new', ordinal=1, date='2027-01-01')
    if corrupt_source:
        # A corrected physical author invalidates the stored Alice attribution.
        # The new endpoint's valid source cannot lend the old endpoint proof.
        with store._relational_connection(write=True, scope='source-correction') as conn:
            conn.execute(f'UPDATE canonical_turns SET sender_actor_id={store._placeholder} WHERE canonical_turn_id={store._placeholder}', ('bob', old_id))
    assert store.set_fact_superseded(old.id, new.id) is not corrupt_source
    audit, = store.get_fact_decisions('owner')
    versions = json.loads(audit['source_versions_json'])
    assert {item[0] for item in versions} == ({new_id, 'segment:segment-new'} if corrupt_source else {old_id, new_id, 'segment:segment-old', 'segment:segment-new'})
    assert all(len(item[1]) == 64 for item in versions)
    assert audit['reason'] == ('unproved_audience' if corrupt_source else 'proved_source_scope')
    assert _saved(store, old.id)['superseded_by'] == (None if corrupt_source else new.id)
    assert json.loads(audit['before_json'])['what'] == old.what
    assert json.loads(audit['after_json'])['what'] == old.what


@pytest.mark.parametrize('action', ['revise', 'supersede'])
def test_audit_insert_failure_rolls_back_fact_and_embedding(store, action):
    old = _fact(store)
    new = _fact(store, 'new', date='2027-01-01')
    store.store_fact_embeddings(old.id, 'owner', 'offline-model', [1.0, 0.0])
    before = _saved(store, old.id)
    if store._relational_dialect == 'postgres':
        import psycopg
        expected_error = psycopg.errors.RaiseException
    else:
        expected_error = sqlite3.IntegrityError
    with _reject_audit_insert(store), pytest.raises(expected_error, match='injected audit insert failure'):
        if action == 'revise':
            store.update_fact_fields(old.id, 'lives in', 'Denver', 'active', 'Alice lives in Denver')
        else:
            store.set_fact_superseded(old.id, new.id)
    assert _saved(store, old.id) == before
    assert store.load_fact_embeddings('owner', 'offline-model')[old.id][1] == [1.0, 0.0]
    assert store.get_fact_decisions('owner') == []
    # The same pooled connection remains usable and commits after rollback.
    assert store.update_fact_fields(old.id, 'lives in', 'Paris', 'active', 'Alice lives in Paris')
    assert store.load_fact_embeddings('owner', 'offline-model') == {}
    assert len(store.get_fact_decisions('owner')) == 1


def _operation(store):
    store.upsert_conversation(tenant_id='tenant', conversation_id='owner')
    epoch = store.get_lifecycle_epoch('owner')
    operation = store.start_compaction_operation(
        conversation_id='owner', lifecycle_epoch=epoch, worker_id='worker', phase_count=1, phase_name='facts',
    )
    with store._relational_connection(write=True, scope='start-test-operation') as conn:
        conn.execute(f"UPDATE compaction_operation SET status='running' WHERE operation_id={store._placeholder}", (operation,))
    return dict(operation_id=operation, owner_worker_id='worker', lifecycle_epoch=epoch)


@pytest.mark.parametrize('action', ['revise', 'supersede'])
@pytest.mark.parametrize('mode,valid', [
    (CompactionFenceMode.ACTIVE, True),
    (CompactionFenceMode.ACTIVE, False),
    (CompactionFenceMode.OBSERVE, False),
])
def test_mutation_guard_checks_exact_worker_and_observe_logs(store, action, mode, valid, caplog):
    old = _fact(store)
    new = _fact(store, 'new', date='2027-01-01')
    guard = _operation(store)
    store._compaction_fence_mode = mode
    if not valid:
        guard['owner_worker_id'] = 'stale-worker'
    def mutate():
        if action == 'revise':
            return store.update_fact_fields(old.id, 'lives in', 'Denver', 'active', '', **guard)
        return store.set_fact_superseded(old.id, new.id, **guard)
    with caplog.at_level(logging.WARNING):
        if not valid and mode is CompactionFenceMode.ACTIVE:
            with pytest.raises(CompactionLeaseLost):
                mutate()
            assert store.get_fact_decisions('owner') == []
            assert _saved(store, old.id)['object'] == 'Boston'
            assert _saved(store, old.id)['superseded_by'] is None
        else:
            assert mutate()
            audit, = store.get_fact_decisions('owner')
            assert audit['operation_id'] == guard['operation_id']
    observed = [record for record in caplog.records if 'COMPACTION_FENCE_OBSERVED_MISMATCH' in record.getMessage()]
    assert len(observed) == (1 if mode is CompactionFenceMode.OBSERVE else 0)
    if observed:
        assert guard['operation_id'] in observed[0].getMessage()
        assert ('update_fact_fields' if action == 'revise' else 'set_fact_superseded') in observed[0].getMessage()


def test_partial_guard_is_rejected_before_mutation(store):
    old = _fact(store)
    with pytest.raises(ValueError, match='all-None or all-non-None'):
        store.update_fact_fields(old.id, 'lives in', 'Denver', 'active', '', operation_id='partial')
    assert _saved(store, old.id)['object'] == 'Boston'
    assert store.get_fact_decisions('owner') == []


@pytest.fixture
def clock(monkeypatch):
    from virtual_context.storage import relational
    value = [1_800_000_000.0]
    monkeypatch.setattr(relational.time, 'time', lambda: value[0])
    return value


def test_direct_delete_tombstone_blocks_old_exchange_until_explicit_recreation(store, clock):
    payload = '{"round":1}'
    assert store.put_pending_exchange('owner', 'exchange', payload, expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'old-claim', now=clock[0]) == payload
    store.delete_conversation('owner')
    assert not store.put_pending_exchange('owner', 'exchange', payload, expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'after-delete', now=clock[0]) is None
    assert not store.finish_pending_exchange('owner', 'exchange', 'old-claim', consume=True)
    with pytest.raises(ConversationLifecycleConflict):
        store.activate_conversation('owner')
    store.activate_conversation('owner', recreate_deleted=True)
    fresh = '{"round":2}'
    assert store.put_pending_exchange('owner', 'exchange', fresh, expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'new-claim', now=clock[0]) == fresh
    assert not store.renew_pending_exchange('owner', 'exchange', 'old-claim', now=clock[0])
    assert not store.finish_pending_exchange('owner', 'exchange', 'old-claim', consume=True)
    assert store.finish_pending_exchange('owner', 'exchange', 'new-claim', consume=True)


def test_owner_epoch_recreation_invalidates_retained_hidden_exchange(store, clock):
    store.upsert_conversation(tenant_id='tenant', conversation_id='owner')
    epoch = store.get_lifecycle_epoch('owner')
    assert store.put_pending_exchange('owner', 'exchange', '{}', expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'old-claim', now=clock[0]) == '{}'
    store.mark_conversation_deleted('owner')
    assert not store.finish_pending_exchange('owner', 'exchange', 'old-claim', consume=True)
    assert store.increment_lifecycle_epoch_on_resurrect('owner') == epoch + 1
    assert store.claim_pending_exchange('owner', 'exchange', 'stale-payload', now=clock[0]) is None
    assert not store.renew_pending_exchange('owner', 'exchange', 'old-claim', now=clock[0])
    assert not store.finish_pending_exchange('owner', 'exchange', 'old-claim', consume=True)
    assert store.put_pending_exchange('owner', 'exchange', '{"new":true}', expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'new-claim', now=clock[0]) == '{"new":true}'


def test_expired_claim_cannot_renew_or_consume_reclaimed_exchange(store, clock):
    assert store.put_pending_exchange('owner', 'exchange', '{}', expires_at=clock[0] + 300)
    assert store.claim_pending_exchange('owner', 'exchange', 'expired', now=clock[0], lease_seconds=1) == '{}'
    clock[0] += 2
    assert not store.renew_pending_exchange('owner', 'exchange', 'expired', now=clock[0])
    assert not store.finish_pending_exchange('owner', 'exchange', 'expired', consume=True)
    assert store.claim_pending_exchange('owner', 'exchange', 'fresh', now=clock[0]) == '{}'
    assert not store.finish_pending_exchange('owner', 'exchange', 'expired', consume=True)
    assert store.finish_pending_exchange('owner', 'exchange', 'fresh', consume=True)
    assert store.list_pending_exchanges('owner', now=clock[0]) == []


def _exchange_rows(store):
    with store._relational_connection() as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM pending_tool_exchanges ORDER BY conversation_id,exchange_id').fetchall()]


@pytest.mark.parametrize('claimed', [False, True])
def test_live_pending_exchange_peek_preserves_existing_claim_and_lease(store, clock, claimed):
    payload = '{"messages":[{"role":"assistant","tool_call":"approved-call"}]}'
    assert store.put_pending_exchange('owner', 'exchange', payload, expires_at=clock[0] + 300)
    if claimed:
        assert store.claim_pending_exchange('owner', 'exchange', 'executor', now=clock[0], lease_seconds=20) == payload
    before = _exchange_rows(store)
    assert before[0]['claim_id'] == ('executor' if claimed else None)
    assert store.get_pending_exchange('owner', 'exchange', now=clock[0] + 1) == payload
    assert store.get_pending_exchange('owner', 'exchange', now=clock[0] + 2) == payload
    assert _exchange_rows(store) == before
    if claimed:
        assert store.claim_pending_exchange('owner', 'exchange', 'interloper', now=clock[0] + 3) is None
    else:
        # Inspection does not acquire a lease that would block execution.
        assert store.claim_pending_exchange('owner', 'exchange', 'executor', now=clock[0] + 3) == payload


@pytest.mark.parametrize('denied', ['expired', 'deleted', 'foreign'])
def test_pending_exchange_peek_denies_invalid_scope_without_mutating_state(store, clock, denied):
    store.upsert_conversation(tenant_id='tenant', conversation_id='owner')
    assert store.put_pending_exchange('owner', 'exchange', '{}', expires_at=clock[0] + 5)
    assert store.claim_pending_exchange('owner', 'exchange', 'executor', now=clock[0]) == '{}'
    if denied == 'deleted':
        store.mark_conversation_deleted('owner')
    before = _exchange_rows(store)
    with store._relational_connection() as conn:
        lifecycle_before = [dict(row) for row in conn.execute('SELECT * FROM conversation_lifecycle ORDER BY conversation_id').fetchall()]
    assert store.get_pending_exchange(
        'foreign' if denied == 'foreign' else 'owner', 'exchange',
        now=clock[0] + (6 if denied == 'expired' else 1),
    ) is None
    assert _exchange_rows(store) == before
    with store._relational_connection() as conn:
        assert [dict(row) for row in conn.execute('SELECT * FROM conversation_lifecycle ORDER BY conversation_id').fetchall()] == lifecycle_before


@pytest.mark.parametrize('changed', ['none', 'old_fact_fields', 'new_fact_fields', 'canonical_text', 'segment_metadata'])
def test_admission_snapshot_cas_rejects_stale_fact_and_source_versions(store, changed):
    old, old_id = _physical_fact(store, 'old')
    new, new_id = _physical_fact(store, 'new', ordinal=1, date='2027-01-01')
    old_snapshot = store.get_fact_admission_snapshot(old.id)
    new_snapshot = store.get_fact_admission_snapshot(new.id)
    assert old_snapshot['audience'] == new_snapshot['audience'] == ('audience', 'channel')
    expected = dict(
        expected_old_version=old_snapshot['fact_version'],
        expected_new_version=new_snapshot['fact_version'],
        expected_source_versions=tuple(sorted(set(old_snapshot['source_versions'] + new_snapshot['source_versions']))),
    )
    assert {key for key, _ in expected['expected_source_versions']} == {
        old_id, new_id, 'segment:segment-old', 'segment:segment-new',
    }
    p = store._placeholder
    if changed != 'none':
        with store._relational_connection(write=True, scope='concurrent-source-correction') as conn:
            if changed.endswith('fact_fields'):
                target = old.id if changed.startswith('old_') else new.id
                conn.execute(f'UPDATE facts SET object={p},what={p} WHERE id={p}', ('Paris', 'Alice clarified she lives in Paris', target))
            elif changed == 'canonical_text':
                conn.execute(f'UPDATE canonical_turns SET user_content={p} WHERE canonical_turn_id={p}', ('Alice corrected the physical source', old_id))
            else:
                metadata = json.loads(conn.execute(f'SELECT metadata_json FROM segments WHERE ref={p}', ('segment-old',)).fetchone()['metadata_json'])
                metadata['session_date'] = '2026-02-01'
                conn.execute(f'UPDATE segments SET metadata_json={p} WHERE ref={p}', (json.dumps(metadata), 'segment-old'))
    before_old, before_new = _saved(store, old.id), _saved(store, new.id)
    observed_old = store.get_fact_admission_snapshot(old.id)
    observed_new = store.get_fact_admission_snapshot(new.id)
    observed_sources = tuple(sorted(set(observed_old['source_versions'] + observed_new['source_versions'])))
    assert store.set_fact_superseded(old.id, new.id, **expected) is (changed == 'none')
    audit, = store.get_fact_decisions('owner')
    assert audit['accepted'] == int(changed == 'none')
    assert audit['reason'] == ('proved_source_scope' if changed == 'none' else 'stale_proposal')
    proposal = json.loads(audit['proposal_json'])
    assert proposal['expected_old_version'] == expected['expected_old_version']
    assert proposal['expected_new_version'] == expected['expected_new_version']
    assert proposal['source_versions'] == [list(pair) for pair in expected['expected_source_versions']]
    assert json.loads(audit['source_versions_json']) == [list(pair) for pair in observed_sources]
    assert json.loads(audit['observed_fact_versions_json']) == {old.id: observed_old['fact_version'], new.id: observed_new['fact_version']}
    assert _saved(store, new.id) == before_new
    if changed != 'none':
        assert _saved(store, old.id) == before_old
        assert json.loads(audit['before_json']) == json.loads(audit['after_json'])
        assert not store.get_fact_links(new.id)
    else:
        assert _saved(store, old.id)['superseded_by'] == new.id


def test_fact_decision_content_is_immutable_but_owner_relocation_preserves_origin(store):
    old, new = _fact(store), _fact(store, 'new', date='2027-01-01')
    assert store.set_fact_superseded(old.id, new.id)
    before, = store.get_fact_decisions('owner')
    assert before['origin_conversation_id'] == 'owner'
    assert set(json.loads(before['observed_fact_versions_json'])) == {old.id, new.id}
    if store._relational_dialect == 'postgres':
        import psycopg
        error = psycopg.errors.RaiseException
    else:
        error = sqlite3.IntegrityError
    with pytest.raises(error, match='immutable'), store._relational_connection(write=True, scope='audit-tamper') as conn:
        conn.execute(f'UPDATE fact_decisions SET reason={store._placeholder} WHERE decision_id={store._placeholder}', ('rewritten', before['decision_id']))
    assert store.get_fact_decisions('owner') == [before]
    with store._relational_connection(write=True, scope='audit-relocation') as conn:
        conn.execute(f'UPDATE fact_decisions SET conversation_id={store._placeholder} WHERE decision_id={store._placeholder}', ('merged-owner', before['decision_id']))
    assert store.get_fact_decisions('owner') == []
    assert store.get_fact_decisions('merged-owner') == [{**before, 'conversation_id': 'merged-owner'}]
