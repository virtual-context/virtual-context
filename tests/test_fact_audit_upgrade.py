"""Upgrade older immutable audit schemas without breaking merge or new fields."""
import sqlite3

import pytest

from tests.test_conversation_unification_admin import _conversation, _merge_source, SOURCE, TARGET
from tests.test_storage_domain_contracts import store as store
from virtual_context.types import Fact


def _old_schema(store):
    """Prepare the actual previous column/trigger shape, retaining its audit."""
    with store._relational_connection(write=True, scope='prepare-old-audit-schema') as conn:
        if store._relational_dialect == 'postgres':
            conn.execute('DROP TRIGGER guard_fact_decision_content ON fact_decisions')
        else:
            conn.execute('DROP TRIGGER guard_fact_decision_content')
        conn.execute('ALTER TABLE fact_decisions DROP COLUMN observed_fact_versions_json')
        conn.execute("UPDATE fact_decisions SET origin_conversation_id=''")
        if store._relational_dialect == 'postgres':
            conn.execute('CREATE TRIGGER guard_fact_decision_content BEFORE UPDATE ON fact_decisions FOR EACH ROW EXECUTE FUNCTION guard_fact_decision_content()')
        else:
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(fact_decisions)')}
            predicate = ' OR '.join(f'NEW.{key} IS NOT OLD.{key}' for key in sorted(columns - {'conversation_id'}))
            conn.execute(f"CREATE TRIGGER guard_fact_decision_content BEFORE UPDATE ON fact_decisions WHEN {predicate} BEGIN SELECT RAISE(ABORT, 'fact decision content is immutable'); END")


def _reopen(store):
    target = store.dsn if store._relational_dialect == 'postgres' else store.db_path
    return type(store)(target, compaction_fence_mode=store._compaction_fence_mode)


def test_legacy_audit_upgrade_real_merge_preserves_unknown_origin_and_allows_delete(store):
    _conversation(store, SOURCE)
    _conversation(store, TARGET)
    store.store_facts([Fact(id='old', conversation_id=SOURCE, subject='Alice', verb='likes', object='tea')])
    assert store.update_fact_fields('old', 'likes', 'coffee', 'active', '')
    _old_schema(store)
    upgraded = _reopen(store)
    try:
        before, = upgraded.get_fact_decisions(SOURCE)
        assert before['origin_conversation_id'] == ''
        assert before['observed_fact_versions_json'] == '{}'
        _merge_source(upgraded)
        assert upgraded.get_fact_decisions(SOURCE) == []
        assert upgraded.get_fact_decisions(TARGET) == [{**before, 'conversation_id': TARGET}]
        assert upgraded.query_facts(conversation_id=TARGET)[0].id == 'old'
        upgraded.delete_conversation(TARGET)
        assert upgraded.get_fact_decisions(TARGET) == []
        assert upgraded.query_facts(conversation_id=TARGET) == []
    finally:
        upgraded.close()


def test_upgrade_guards_added_observed_version_column_and_keeps_payload_immutable(store):
    store.store_facts([Fact(id='fact', conversation_id='owner', subject='Alice')])
    assert store.update_fact_fields('fact', 'likes', 'tea', 'active', '')
    _old_schema(store)
    upgraded = _reopen(store)
    try:
        before, = upgraded.get_fact_decisions('owner')
        if upgraded._relational_dialect == 'postgres':
            import psycopg
            error = psycopg.errors.RaiseException
        else:
            error = sqlite3.IntegrityError
        with pytest.raises(error, match='immutable'), upgraded._relational_connection(write=True, scope='tamper-added-column') as conn:
            conn.execute(f'UPDATE fact_decisions SET observed_fact_versions_json={upgraded._placeholder}', ('{"forged":"version"}',))
        assert upgraded.get_fact_decisions('owner') == [before]
        if upgraded._relational_dialect == 'sqlite':
            statements = []
            conn = upgraded._get_conn()
            conn.set_trace_callback(statements.append)
            upgraded._ensure_request_state_schema()
            conn.set_trace_callback(None)
            assert not any(sql.lstrip().startswith(('CREATE TRIGGER','DROP TRIGGER')) for sql in statements)
    finally:
        upgraded.close()
