import json
from datetime import datetime, timezone

from virtual_context.config import VirtualContextConfig
from virtual_context.core.compactor import DomainCompactor
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import (
    Fact,
    RetrieverConfig,
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    TagGeneratorConfig,
)


GUILD = "sk:agent:bast:discord:channel:15249"
OPTICS = "actor:discord:111"
BIGTEX = "actor:discord:222"


class _NoopEmbeddingProvider:
    @staticmethod
    def get_embed_fn():
        return None


def _engine(tmp_path):
    config = VirtualContextConfig(
        conversation_id=GUILD,
        tenant_id="t1",
        storage=StorageConfig(
            backend="sqlite", sqlite_path=str(tmp_path / "backfill.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
        retriever=RetrieverConfig(inbound_tagger_type="disabled"),
    )
    engine = VirtualContextEngine(
        config=config, embedding_provider=_NoopEmbeddingProvider(),
    )
    engine._store.upsert_conversation(tenant_id="t1", conversation_id=GUILD)
    return engine


def _conversation_info(sender_id, message_id, *, reply_to_id=""):
    payload = {
        "sender_id": sender_id,
        "sender": "Optics" if sender_id == "111" else "BigTex",
        "message_id": message_id,
        "chat_id": "channel:15249",
    }
    if reply_to_id:
        payload["reply_to_id"] = reply_to_id
    return (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(payload) + "\n```\n"
    )


def _reply_target():
    return (
        "\nReply target of current user message (context):\n```json\n"
        + json.dumps({
            "sender_id": "222", "sender_label": "BigTex",
            "message_id": "m1", "body": "protocol claim",
        })
        + "\n```\n"
    )


def test_backfill_actors_uses_only_provider_effective_text(tmp_path):
    engine = _engine(tmp_path)
    raw = json.dumps([
        {"type": "text", "text": _conversation_info("222", "old") + "old"},
        {"type": "text", "text": _conversation_info("111", "m2") + "hello"},
    ])
    engine._store.save_canonical_turn(
        GUILD, -1, "hello", "", canonical_turn_id="ct2", sort_key=1,
        turn_hash="h2", user_raw_content=raw,
    )

    report = engine.backfill_actors(GUILD)
    row = engine._store.get_all_canonical_turns(GUILD)[0]

    assert report["updated"] == 1
    assert row.sender_actor_id == OPTICS
    assert engine._store.get_actor_profile("t1", OPTICS) is not None
    engine.close()


def test_backfill_reply_roles_resolves_exact_same_audience_target(tmp_path):
    engine = _engine(tmp_path)
    engine._store.save_canonical_turn(
        GUILD, -1, "protocol claim", "", canonical_turn_id="target",
        sort_key=1, turn_hash="target", sender="BigTex",
        sender_actor_id=BIGTEX, source_message_id="m1",
        audience_conversation_id=GUILD, audience_attribution_version=1,
        origin_channel_id="15249",
    )
    raw_text = (
        _conversation_info("111", "m2", reply_to_id="m1")
        + "thoughts?" + _reply_target()
    )
    engine._store.save_canonical_turn(
        GUILD, -1, "thoughts?", "", canonical_turn_id="reply",
        sort_key=2, turn_hash="reply", user_raw_content=json.dumps([
            {"type": "text", "text": raw_text},
        ]),
    )

    report = engine.backfill_reply_roles(GUILD)
    reply = next(
        row for row in engine._store.get_all_canonical_turns(GUILD)
        if row.canonical_turn_id == "reply"
    )

    assert report["updated"] == 1
    assert reply.reply_subject_actor_id == BIGTEX
    assert reply.reply_target_body == "protocol claim"
    assert reply.audience_conversation_id == GUILD
    engine.close()


def test_backfill_fact_authors_redistills_with_closed_roster(tmp_path):
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    engine._store.save_canonical_turn(
        GUILD, -1, "I prefer terse answers", "noted",
        canonical_turn_id="ct", sort_key=1, turn_hash="ct",
        sender="Optics", sender_actor_id=OPTICS,
        audience_conversation_id=GUILD, audience_attribution_version=1,
    )
    engine._store.store_segment(StoredSegment(
        ref="seg", conversation_id=GUILD, primary_tag="prefs",
        tags=["prefs"], summary="old", full_text="old", messages=[],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct"], source_mapping_complete=True,
            turn_count=1,
        ),
        created_at=now, start_timestamp=now, end_timestamp=now,
    ))
    engine._store.store_facts([Fact(
        id="old", subject="user", verb="prefers", object="terse answers",
        segment_ref="seg", conversation_id=GUILD,
    )])

    class LLM:
        def complete(self, **_kwargs):
            return json.dumps({
                "summary": "Optics prefers terse answers.",
                "entities": [], "key_decisions": [], "action_items": [],
                "date_references": [], "refined_tags": ["prefs"],
                "facts": [{
                    "subject": "Optics", "verb": "prefers",
                    "object": "terse answers", "status": "active",
                    "fact_type": "personal", "what": "Optics prefers terse answers.",
                    "speaker": "Optics",
                }],
            }), {}

    engine._compactor = DomainCompactor(LLM(), engine.config.compactor)
    engine._compaction._compactor = engine._compactor
    report = engine.backfill_fact_authors(GUILD)
    facts = engine._store.get_facts_by_segment("seg")

    assert report["updated"] == 1
    assert len(facts) == 1
    assert facts[0].author_actor_id == OPTICS
    engine.close()


def test_version_one_with_empty_author_is_reattempted_not_skipped(tmp_path):
    """A failed attribution must not masquerade as a finished one.

    A sole-actor (version 1) fact is stamped whether or not the model's
    speaker label resolved to an actor; when the physical rows carried no
    actor ids at distill time it resolves to nothing and the fact keeps
    version 1 with an EMPTY author. That is not "already attributed" — it is
    an attribution that failed and can now succeed once the rows gain actor
    ids. The backfill must reconsider it, not skip it as existing.
    """
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    engine._store.save_canonical_turn(
        GUILD, -1, "I prefer terse answers", "noted",
        canonical_turn_id="ct-empty", sort_key=1, turn_hash="ct-empty",
        sender="Optics", sender_actor_id=OPTICS,
        audience_conversation_id=GUILD, audience_attribution_version=1,
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-empty", conversation_id=GUILD, primary_tag="prefs",
        tags=["prefs"], summary="s", full_text="f", messages=[],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-empty"], source_mapping_complete=True,
            turn_count=1,
        ),
        created_at=now, start_timestamp=now, end_timestamp=now,
    ))
    # version 1, but the author never resolved — the failure case.
    engine._store.store_facts([Fact(
        id="unresolved", subject="user", verb="prefers",
        object="terse answers", segment_ref="seg-empty",
        conversation_id=GUILD,
        author_actor_id="", author_attribution_version=1,
        author_source_role="unattributed",
    )])

    class LLM:
        def complete(self, **_kwargs):
            return json.dumps({
                "summary": "Optics prefers terse answers.",
                "entities": [], "key_decisions": [], "action_items": [],
                "date_references": [], "refined_tags": ["prefs"],
                "facts": [{
                    "subject": "Optics", "verb": "prefers",
                    "object": "terse answers", "status": "active",
                    "fact_type": "personal",
                    "what": "Optics prefers terse answers.",
                    "speaker": "Optics",
                }],
            }), {}

    engine._compactor = DomainCompactor(LLM(), engine.config.compactor)
    engine._compaction._compactor = engine._compactor
    report = engine.backfill_fact_authors(GUILD)

    assert report["skipped_existing"] == 0, (
        "a version-1 fact with no author was wrongly treated as attributed"
    )
    assert report["updated"] == 1
    facts = engine._store.get_facts_by_segment("seg-empty")
    assert facts[0].author_actor_id == OPTICS
    engine.close()


def test_reply_lane_facts_are_settled_and_skipped(tmp_path):
    """A version-2 reply-lane fact is done — even an empty author is correct
    for an assistant lane — so re-running must not re-spend on it."""
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    engine._store.save_canonical_turn(
        GUILD, -1, "the whey is 30g", "noted",
        canonical_turn_id="ct-settled", sort_key=1, turn_hash="ct-settled",
        sender="BigTex", sender_actor_id=BIGTEX,
        audience_conversation_id=GUILD, audience_attribution_version=1,
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-settled", conversation_id=GUILD, primary_tag="macros",
        tags=["macros"], summary="s", full_text="f", messages=[],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-settled"], source_mapping_complete=True,
            turn_count=1,
        ),
        created_at=now, start_timestamp=now, end_timestamp=now,
    ))
    engine._store.store_facts([Fact(
        id="settled", subject="whey", verb="is", object="30g",
        segment_ref="seg-settled", conversation_id=GUILD,
        author_actor_id=BIGTEX, author_attribution_version=2,
        author_source_role="requester",
    )])

    class LLM:
        def complete(self, **_kwargs):
            raise AssertionError("re-distill must not run for a settled fact")

    engine._compactor = DomainCompactor(LLM(), engine.config.compactor)
    engine._compaction._compactor = engine._compactor
    report = engine.backfill_fact_authors(GUILD)

    assert report["skipped_existing"] == 1
    assert report["updated"] == 0
    engine.close()


def test_rebuild_actor_cards_runs_while_read_gate_is_dark(tmp_path):
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    engine._store.save_canonical_turn(
        GUILD, -1, "I prefer terse answers", "noted",
        canonical_turn_id="ct-card", sort_key=1, turn_hash="ct-card",
        sender="Optics", sender_actor_id=OPTICS,
        audience_conversation_id=GUILD, audience_attribution_version=1,
        origin_channel_id="15249",
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-card", conversation_id=GUILD, primary_tag="prefs",
        tags=["prefs"], summary="s", full_text="f", messages=[],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-card"], source_mapping_complete=True,
        ),
        created_at=now, start_timestamp=now, end_timestamp=now,
    ))
    engine._store.store_facts([Fact(
        id="fact-card", subject="Optics", verb="prefers",
        object="terse answers", what="Optics prefers terse answers.",
        segment_ref="seg-card", conversation_id=GUILD,
        author_actor_id=OPTICS, author_attribution_version=1,
        author_source_role="requester",
    )])
    engine._store.upsert_actor_profile_from_turn(
        GUILD, OPTICS, "Optics", seen_at=now.isoformat(),
    )

    class LLM:
        def complete(self, **_kwargs):
            return json.dumps({
                "substantive": True,
                "coverage_reason": "substantive",
                "entries": [{
                        "kind": "communication_pref",
                        "body": "prefers terse answers",
                        "confidence": 0.9,
                        "fact_ids": ["fact-card"],
                    "turn_ids": [],
                }],
            }), {}

    class Admission:
        def complete(self, **kwargs):
            candidate = json.loads(kwargs["user"])["candidates"][0]
            return json.dumps({
                "substantive": True,
                "coverage_reason": "substantive",
                "decisions": [{
                        "candidate_id": candidate["candidate_id"],
                        "admit": True,
                        "reason": "durable",
                }],
            }), {}

    engine._compactor = DomainCompactor(LLM(), engine.config.compactor)
    engine._compaction._compactor = engine._compactor
    engine.config.assembler.actor_card_admission_model = "semantic-model"
    engine._compaction._actor_card_admission_provider_override = Admission()
    assert engine.config.assembler.actor_card_enabled is False

    report = engine.rebuild_actor_cards(GUILD)
    card = engine._store.get_actor_card(
        "t1", OPTICS, owner_conversation_id=GUILD,
        audience_conversation_id=GUILD, audience_channel_id="15249",
    )

    assert report == {"eligible": 1, "rebuilt": 1, "failed": 0, "dry_run": False}
    assert card is not None
    assert [entry.body for entry in card.entries] == ["prefers terse answers"]
    engine.close()


def test_batch_operation_continues_past_a_vanished_conversation(tmp_path, capsys, monkeypatch):
    """A conversation deleted between enumeration and processing must not
    abort the rest of the batch."""
    import json as _json
    import sys as _sys
    from types import SimpleNamespace
    from virtual_context.cli import main as cli_main

    calls = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]
            self._store = SimpleNamespace(
                list_canonical_conversation_ids=lambda **kw: ["conv-a", "conv-gone", "conv-b"],
            )

        def backfill_reply_roles(self, target, *, dry_run, limit):
            calls.append(target)
            if target == "conv-gone":
                raise KeyError(target)
            return {"eligible": 1, "updated": 1, "failed": 0}

        def close(self):
            pass

    monkeypatch.setattr(
        "virtual_context.engine.VirtualContextEngine", _FakeEngine,
    )
    args = SimpleNamespace(
        conversation_id="", tenant_id="", all_convs_for_tenant=True,
        dry_run=False, limit=None, config=None,
        storage_backend="sqlite", postgres_dsn=None,
        sqlite_path=str(tmp_path / "s.db"),
    )
    cli_main._cmd_admin_actor_operation(
        args, "backfill_reply_roles", ("eligible", "updated", "failed"),
    )
    out = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "ok"
    assert out["errors"] == 1
    assert out["updated"] == 2
    assert calls == ["conv-a", "conv-gone", "conv-b"]
    error_rows = [r for r in out["results"] if r.get("status") == "error"]
    assert len(error_rows) == 1 and error_rows[0]["conversation_id"] == "conv-gone"
