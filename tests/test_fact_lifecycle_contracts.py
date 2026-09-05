"""Evidence and chronology remain invariant across supersession modes."""
from unittest.mock import Mock, patch

import pytest

from virtual_context.ingest.supersession import (
    FactLinkChecker,
    FactSupersessionChecker,
    promote_planned_facts,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, SupersessionConfig


@pytest.fixture
def store(tmp_path):
    db = SQLiteStore(tmp_path / 'facts.db')
    yield db
    db.close()


def test_elapsed_plan_preserves_source_status_text_and_embedding(store):
    fact = Fact(id='plan', subject='Alice', verb='plans to attend', object='conference',
                what='Alice plans to attend a conference', status='planned',
                when_date='2020-01-01', conversation_id='conversation')
    store.store_facts([fact])
    store.store_fact_embeddings('plan', 'conversation', 'model', [1.0, 0.0])
    provider = Mock()
    assert promote_planned_facts(
        store, reference_date='2026-09-05', llm_provider=provider,
        conversation_id='conversation', operation_id='old-operation',
        owner_worker_id='old-worker', lifecycle_epoch=0,
    ) == 0
    provider.complete.assert_not_called()
    saved = store.query_facts(conversation_id='conversation')[0]
    assert (saved.status, saved.verb, saved.what) == (fact.status, fact.verb, fact.what)
    assert store.load_fact_embeddings('conversation', 'model')['plan'][1] == [1.0, 0.0]


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('incoming_date,expected_count', [('2025-01-01', 0), ('2027-01-01', 1)])
def test_supersession_modes_share_chronology_admission(store, graph, incoming_date, expected_count):
    current = Fact(id='current', subject='Alice', verb='lives in', object='Boston',
                   session_date='2026-01-01', conversation_id='conversation')
    incoming = Fact(id='incoming', subject='Alice', verb='lives in', object='Denver',
                    session_date=incoming_date, conversation_id='conversation')
    store.store_facts([current, incoming])
    provider = Mock()
    provider.complete.return_value = (
        '{"superseded":[0],"links":[{"source":"N0","target":"E0",'
        '"relation":"supersedes"}]}' if graph else '[0]', {},
    )
    provider.last_usage = {}
    kwargs = dict(llm_provider=provider, model='fake', store=store,
                  config=SupersessionConfig(enabled=True))
    if graph:
        checker = FactLinkChecker(**kwargs, graph_links=True)
        links, superseded = checker.check_and_link([incoming], conversation_id='conversation')
        assert links == expected_count
        assert superseded == expected_count
        assert len(store.get_fact_links('incoming')) == expected_count
    else:
        checker = FactSupersessionChecker(**kwargs)
        assert checker.check_and_supersede([incoming]) == expected_count
    remaining = {f.id for f in store.query_facts(conversation_id='conversation')}
    assert ('current' in remaining) is (expected_count == 0)


def _source_fact(store, fact_id, *, author='actor-A', audience='audience-A',
                 channel='channel-A', role='requester', version=2,
                 date='2026-01-01', what='Alice lives in Boston'):
    from virtual_context.types import SegmentMetadata, StoredSegment

    canonical_id = f'row-{fact_id}'
    store.save_canonical_turn(
        'conversation', -1, what, '', canonical_turn_id=canonical_id,
        turn_group_number=0, turn_hash=canonical_id,
        primary_tag='location', tags=['location'], sender_actor_id=author,
        audience_conversation_id=audience, audience_attribution_version=1,
        origin_channel_id=channel, source_message_id=f"message-{fact_id}",
    )
    segment_ref = f'segment-{fact_id}'
    store.store_segment(StoredSegment(
        ref=segment_ref, conversation_id='conversation', primary_tag='location',
        tags=['location'], summary=what, full_text=what,
        metadata=SegmentMetadata(canonical_turn_ids=[canonical_id], source_mapping_complete=True),
    ))
    fact = Fact(id=fact_id, subject='Alice', verb='lives in', object='Boston',
                what=what, conversation_id='conversation', tags=['location'],
                session_date=date, segment_ref=segment_ref,
                author_actor_id=author, author_attribution_version=version,
                author_source_role=role, author_source_message_id=f"message-{fact_id}" if version == 2 else "")
    store.store_facts([fact])
    return fact


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('mismatch', ['author', 'audience', 'channel', 'role', 'date', 'unproved', 'conversation'])
def test_scope_and_chronology_reject_before_any_model_sees_candidate(store, graph, mismatch):
    old_args = {}
    if mismatch in ('author', 'audience', 'channel', 'role'):
        old_args[mismatch] = {'author': 'actor-B', 'audience': 'audience-B', 'channel': 'channel-B', 'role': 'subject'}[mismatch]
    if mismatch == 'date':
        old_args['date'] = '2028-01-01'
    current = _source_fact(store, 'old', **old_args)
    incoming = _source_fact(store, 'new', date='2027-01-01', what='Alice lives in Denver')
    if mismatch == 'unproved':
        current.segment_ref = 'missing-segment'
        store.store_facts([current])
    if mismatch == 'conversation':
        current.conversation_id = 'foreign-conversation'
        store.store_facts([current])
    provider = Mock()
    provider.complete.side_effect = AssertionError('ineligible candidate reached the model')
    kwargs = dict(llm_provider=provider, model='fake', store=store, config=SupersessionConfig(enabled=True))
    if graph:
        assert FactLinkChecker(**kwargs, graph_links=True).check_and_link([incoming], conversation_id='conversation') == (0, 0)
    else:
        assert FactSupersessionChecker(**kwargs).check_and_supersede([incoming]) == 0
    provider.complete.assert_not_called()


@pytest.mark.parametrize('graph', [False, True])
def test_valid_attributed_supersession_records_versions_without_rewriting_source(store, graph):
    import json

    current = _source_fact(store, 'old')
    incoming = _source_fact(store, 'new', date='2027-01-01', what='Alice lives in Denver')
    provider = Mock()
    provider.last_usage = {}
    provider.complete.return_value = ('{"superseded":[0],"links":[{"source":"N0","target":"E0","relation":"supersedes"}]}' if graph else '[0]', {})
    kwargs = dict(llm_provider=provider, model='fake', store=store, config=SupersessionConfig(enabled=True))
    if graph:
        assert FactLinkChecker(**kwargs, graph_links=True).check_and_link([incoming], conversation_id='conversation') == (1, 1)
    else:
        assert FactSupersessionChecker(**kwargs).check_and_supersede([incoming]) == 1
    decision = store.get_fact_decisions('conversation')[0]
    assert decision['accepted'] == 1
    assert {item[0] for item in json.loads(decision['source_versions_json'])} == {'row-old', 'row-new', 'segment:segment-old', 'segment:segment-new'}
    assert json.loads(decision['before_json'])['what'] == current.what
    assert json.loads(decision['after_json'])['what'] == current.what
    provider.complete.reset_mock()
    FactSupersessionChecker(**kwargs)._merge_facts(incoming, current)
    provider.complete.assert_not_called()
    assert store.query_facts(conversation_id='conversation')[0].what == incoming.what


def test_dedup_never_crosses_conversation_or_attributed_source_scope(store):
    from virtual_context.ingest.supersession import dedup_facts

    first = Fact(id='first', subject='Alice', verb='likes', object='tea', what='Alice likes tea', conversation_id='one', session_date='2025-01-01')
    duplicate = Fact(id='duplicate', subject='Alice', verb='likes', object='tea', what='Alice likes tea', conversation_id='one', session_date='2026-01-01')
    foreign = Fact(id='foreign', subject='Alice', verb='likes', object='tea', what='Alice likes tea', conversation_id='two')
    store.store_facts([first, duplicate, foreign])
    assert dedup_facts(store) == 1
    assert {fact.id for fact in store.query_facts(limit=10)} == {'duplicate', 'foreign'}
    _source_fact(store, 'source-A', author='actor-A')
    _source_fact(store, 'source-B', author='actor-B')
    assert dedup_facts(store, conversation_id='conversation') == 0


@pytest.mark.parametrize('graph', [False, True])
def test_rejected_atomic_write_does_not_count_or_publish_supersession_link(store, graph):
    from unittest.mock import patch

    old = Fact(id='old', subject='Alice', verb='likes', object='tea', conversation_id='conversation')
    new = Fact(id='new', subject='Alice', verb='likes', object='coffee', conversation_id='conversation')
    store.store_facts([old, new])
    provider = Mock()
    provider.last_usage = {}
    provider.complete.return_value = ('{"superseded":[0],"links":[{"source":"N0","target":"E0","relation":"supersedes"}]}' if graph else '[0]', {})
    kwargs = dict(llm_provider=provider, model='fake', store=store, config=SupersessionConfig(enabled=True))
    with patch.object(store, 'set_fact_superseded', return_value=False):
        if graph:
            assert FactLinkChecker(**kwargs, graph_links=True).check_and_link([new], conversation_id='conversation') == (0, 0)
        else:
            assert FactSupersessionChecker(**kwargs).check_and_supersede([new]) == 0
    assert not store.get_fact_links('new')


def test_proposals_are_immutable_and_a_plan_cannot_replace_observed_state():
    from dataclasses import FrozenInstanceError
    from virtual_context.core.fact_lifecycle import FactProposal, decide_supersession

    proposal = FactProposal('supersede', 'old', 'new', source_versions=(('row', 'digest'),))
    with pytest.raises(FrozenInstanceError):
        proposal.action = 'revise'
    with pytest.raises(TypeError):
        FactProposal('supersede', 'old', 'new', source_versions=[('row', 'digest')])
    old = Fact(id='old', subject='Alice', status='active', conversation_id='conversation')
    new = Fact(id='new', subject='Alice', status='planned', conversation_id='conversation', when_date='2099-01-01')
    assert decide_supersession(new, old).reason == 'plan_is_not_observed_outcome'


@pytest.mark.parametrize('change', ['actor', 'duplicate', 'missing_id', 'assistant', 'foreign'])
def test_requester_author_proof_requires_current_unique_physical_source(change):
    from dataclasses import replace
    from virtual_context.core.fact_lifecycle import source_author_matches

    fact = Fact(id='fact', conversation_id='conversation', author_actor_id='actor-A',
                author_attribution_version=2, author_source_role='requester', author_source_message_id='message-A')
    row = {'conversation_id': 'conversation', 'user_content': 'source', 'sender_actor_id': 'actor-A', 'source_message_id': 'message-A'}
    assert source_author_matches(fact, [row])
    rows = [dict(row)]
    if change == 'actor':
        rows[0]['sender_actor_id'] = 'actor-B'
    elif change == 'duplicate':
        rows.append(dict(row))
    elif change == 'missing_id':
        fact = replace(fact, author_source_message_id='')
    elif change == 'assistant':
        fact = replace(fact, author_source_role='assistant')
    else:
        rows[0]['conversation_id'] = 'foreign'
    assert not source_author_matches(fact, rows)


@pytest.mark.parametrize('change', ['subject_actor', 'quote', 'version', 'duplicate', 'physical_target'])
def test_subject_proof_uses_reply_target_without_requester_substitution(change):
    from virtual_context.core.fact_lifecycle import source_author_matches

    fact = Fact(id='fact', conversation_id='conversation', author_actor_id='subject-A',
                author_attribution_version=2, author_source_role='subject', author_source_message_id='quoted-message')
    row = {'conversation_id': 'conversation', 'user_content': 'a question', 'sender_actor_id': 'requester-B', 'source_message_id': 'request-message', 'reply_target_message_id': 'quoted-message', 'reply_target_body': 'subject source text', 'reply_subject_actor_id': 'subject-A', 'reply_attribution_version': 1, 'audience_conversation_id': 'audience', 'origin_channel_id': 'channel'}
    assert source_author_matches(fact, [row])
    rows = [dict(row)]
    if change == 'subject_actor':
        rows[0]['reply_subject_actor_id'] = 'requester-B'
    elif change == 'quote':
        rows[0]['reply_target_body'] = ''
    elif change == 'version':
        rows[0]['reply_attribution_version'] = 0
    elif change == 'duplicate':
        rows.append(dict(row))
    else:
        rows.append({**row, 'source_message_id': 'quoted-message', 'sender_actor_id': 'subject-A'})
    assert not source_author_matches(fact, rows)


def test_sole_actor_proof_rejects_mixed_unknown_and_reply_rosters():
    from virtual_context.core.fact_lifecycle import source_author_matches

    fact = Fact(conversation_id='conversation', author_actor_id='actor-A', author_attribution_version=1, author_source_role='requester')
    row = {'conversation_id': 'conversation', 'user_content': 'source', 'sender_actor_id': 'actor-A'}
    assert source_author_matches(fact, [row, dict(row)])
    for second in ({**row, 'sender_actor_id': ''}, {**row, 'sender_actor_id': 'actor-B'}, {**row, 'reply_target_body': 'unproved quote'}):
        assert not source_author_matches(fact, [row, second])


@pytest.mark.parametrize('changed', ['fact', 'canonical', 'segment'])
@pytest.mark.parametrize('graph', [False, True])
def test_model_proposal_cannot_supersede_a_source_changed_during_comparison(store, changed, graph):
    import json

    _source_fact(store, 'old')
    incoming = _source_fact(store, 'new', date='2027-01-01', what='Alice lives in Denver')
    provider = Mock()
    provider.last_usage = {}
    def compare(**kwargs):
        if changed == 'fact':
            store.update_fact_fields('old', 'lives in', 'Paris', 'active', 'Alice clarified that she lives in Paris')
        else:
            conn = store._get_conn()
            if changed == 'canonical':
                conn.execute("UPDATE canonical_turns SET user_content='corrected source' WHERE canonical_turn_id='row-old'")
            else:
                row = conn.execute("SELECT metadata_json FROM segments WHERE ref='segment-old'").fetchone()
                metadata = json.loads(row[0])
                metadata['session_date'] = '2026-02-01'
                conn.execute("UPDATE segments SET metadata_json=? WHERE ref='segment-old'", (json.dumps(metadata),))
            conn.commit()
        return ('{"superseded":[0],"links":[{"source":"N0","target":"E0","relation":"supersedes"}]}' if graph else '[0]', {})
    provider.complete.side_effect = compare
    kwargs = dict(llm_provider=provider, model='fake', store=store, config=SupersessionConfig(enabled=True))
    if graph:
        assert FactLinkChecker(**kwargs, graph_links=True).check_and_link([incoming], conversation_id='conversation') == (0, 0)
    else:
        assert FactSupersessionChecker(**kwargs).check_and_supersede([incoming]) == 0
    decisions = [item for item in store.get_fact_decisions('conversation') if item['action'] == 'supersede']
    assert decisions and decisions[0]['accepted'] == 0
    assert decisions[0]['reason'] == 'stale_proposal'
    assert not store.get_fact_links('new')


@pytest.mark.parametrize('mode', ['direct', 'embedding', 'graph'])
def test_candidate_admission_reads_each_snapshot_once_and_preserves_post_model_cas(store, mode):
    old = _source_fact(store, 'old')
    incoming = _source_fact(store, 'new', what='Alice lives in Denver')
    if mode == 'embedding':
        # Force discovery through the embedding pool rather than tag/object SQL.
        incoming.tags = ['new-location']
        incoming.object = 'Denver'
        store.store_facts([incoming])
    provider = Mock()
    provider.last_usage = {}

    def compare(**kwargs):
        store._get_conn().execute(
            "UPDATE canonical_turns SET user_content='clarified evidence' WHERE canonical_turn_id='row-old'",
        )
        return ('{"superseded":[0],"links":[{"source":"N0","target":"E0","relation":"supersedes"}]}' if mode == 'graph' else '[0]', {})

    provider.complete.side_effect = compare
    embedding = Mock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
    kwargs = dict(llm_provider=provider, model='fake', store=store,
                  config=SupersessionConfig(enabled=True), embed_fn=embedding if mode == 'embedding' else None)
    with patch.object(store, 'get_fact_admission_snapshot', wraps=store.get_fact_admission_snapshot) as snapshots, patch.object(
        store, 'get_fact_admission_scope', side_effect=AssertionError('redundant source proof read'),
    ):
        if mode == 'graph':
            assert FactLinkChecker(**kwargs, graph_links=True).check_and_link([incoming], conversation_id='conversation') == (0, 0)
        else:
            assert FactSupersessionChecker(**kwargs).check_and_supersede([incoming]) == 0
        assert sorted(call.args[0] for call in snapshots.call_args_list) == [incoming.id, old.id]
    provider.complete.assert_called_once()
    if mode == 'embedding':
        embedding.assert_called()
    assert store.get_fact_decisions('conversation')[0]['reason'] == 'stale_proposal'
    assert {fact.id for fact in store.query_facts(conversation_id='conversation')} == {'old', 'new'}
    assert store.get_fact_links('new') == []


@pytest.mark.parametrize('column', ['updated_at', 'last_seen_at', 'tagged_at', 'compacted_at', 'compaction_operation_id'])
def test_source_maintenance_does_not_invalidate_admitted_proposal(store, column):
    _source_fact(store, 'old')
    _source_fact(store, 'new', date='2027-01-01')
    old = store.get_fact_admission_snapshot('old')
    new = store.get_fact_admission_snapshot('new')
    with store._relational_connection(write=True) as conn:
        conn.execute(f'UPDATE canonical_turns SET {column}=? WHERE canonical_turn_id=?',
                     ('maintenance-marker', 'row-old'))
    assert store.get_fact_admission_snapshot('old') == old
    assert store.set_fact_superseded(
        'old', 'new', expected_old_version=old['fact_version'], expected_new_version=new['fact_version'],
        expected_source_versions=tuple(sorted(set(old['source_versions'] + new['source_versions']))),
    )


def test_rejected_proposal_audits_expected_and_observed_source_versions(store):
    import json
    _source_fact(store, 'old')
    _source_fact(store, 'new', date='2027-01-01')
    old = store.get_fact_admission_snapshot('old')
    new = store.get_fact_admission_snapshot('new')
    expected = tuple(sorted(set(old['source_versions'] + new['source_versions'])))
    with store._relational_connection(write=True) as conn:
        conn.execute("UPDATE canonical_turns SET user_content='corrected source' WHERE canonical_turn_id='row-old'")
    observed = store.get_fact_admission_snapshot('old')
    assert not store.set_fact_superseded(
        'old', 'new', expected_old_version=old['fact_version'], expected_new_version=new['fact_version'],
        expected_source_versions=expected,
    )
    audit, = store.get_fact_decisions('conversation')
    assert audit['origin_conversation_id'] == 'conversation'
    assert dict(json.loads(audit['proposal_json'])['source_versions']) == dict(expected)
    assert dict(json.loads(audit['source_versions_json'])) == dict(observed['source_versions'] + new['source_versions'])
    assert dict(json.loads(audit['source_versions_json'])) != dict(expected)
    assert json.loads(audit['observed_fact_versions_json']) == {'old': old['fact_version'], 'new': new['fact_version']}
