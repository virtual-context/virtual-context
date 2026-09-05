"""Explicit PostgreSQL maintenance that must not run during worker startup."""

_READ_INDEX = 'idx_canonical_turn_group_read'
_READ_INDEX_LOCK = 0x7663526561644964


def migrate_bounded_read_indexes(conn, *, dry_run=True):
    """Build the canonical group index concurrently on an idle connection."""
    if type(dry_run) is not bool:
        raise TypeError('dry_run must be a boolean')
    def existing():
        return conn.execute("""SELECT i.indisvalid, i.indisready,
            pg_get_indexdef(i.indexrelid,1,true) AS first_column,
            pg_get_indexdef(i.indexrelid,2,true) AS second_column,
            pg_get_indexdef(i.indexrelid,3,true) AS third_column,
            i.indnkeyatts, i.indpred IS NULL AS unconditional,
            t.relname AS table_name
            FROM pg_index i JOIN pg_class t ON t.oid=i.indrelid
            WHERE i.indexrelid=to_regclass(%s)""", ('public.' + _READ_INDEX,)).fetchone()
    row = existing()
    status = 'ok' if row and row['indisvalid'] and row['indisready'] else 'needs_build'
    if row and (row['table_name'] != 'canonical_turns' or row['indnkeyatts'] != 3
                or not row['unconditional'] or
                tuple(row[key] for key in ('first_column','second_column','third_column'))
                != ('conversation_id','turn_group_number','sort_key')):
        raise RuntimeError('Canonical group index has an incompatible definition')
    report = {'dry_run': dry_run, 'indexes': {_READ_INDEX: status}}
    if dry_run or status == 'ok':
        return report
    if not conn.autocommit or conn.info.transaction_status != 0:
        raise RuntimeError('Read-index migration requires an idle autocommit connection')
    lock = conn.execute('SELECT pg_try_advisory_lock(%s) AS acquired', (_READ_INDEX_LOCK,)).fetchone()
    if not lock['acquired']:
        raise RuntimeError('Another read-index migration is running')
    previous_timeout = conn.execute('SHOW lock_timeout').fetchone()['lock_timeout']
    try:
        conn.execute("SELECT set_config('lock_timeout', '2s', FALSE)")
        row = existing()
        if row and not (row['indisvalid'] and row['indisready']):
            conn.execute(f'DROP INDEX CONCURRENTLY public.{_READ_INDEX}')
        conn.execute(f'CREATE INDEX CONCURRENTLY IF NOT EXISTS {_READ_INDEX} '
                     'ON public.canonical_turns (conversation_id,turn_group_number,sort_key)')
        row = existing()
        if not row or not row['indisvalid'] or not row['indisready']:
            raise RuntimeError('Canonical group index did not become valid')
        report['indexes'][_READ_INDEX] = 'built'
        return report
    finally:
        try:
            conn.execute("SELECT set_config('lock_timeout', %s, FALSE)", (previous_timeout,))
        finally:
            conn.execute('SELECT pg_advisory_unlock(%s)', (_READ_INDEX_LOCK,))
