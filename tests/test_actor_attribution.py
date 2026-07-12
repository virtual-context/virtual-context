"""Actor identity and the reply edge across the canonical-turn write path.

The write half of actor-aware retrieval. ``sender`` stores a display *name*,
which changes and collides; ``sender_actor_id`` stores the platform account
behind the message, which does not. The reply edge stores who was quoted, in a
lane structurally separate from the requester's own words.

These tests pin:

* ``ingest_batch`` derives the actor from already-parsed ``Message.metadata``
  plus the RAW caller key, whose platform segment survives alias resolution.
* An assistant row is never newly labeled with a human actor or reply edge.
* Preservation is one-way: a resend without the envelope cannot blank a
  stored actor or edge.
* The overlap fast-skip upgrades an empty stored value through the
  epoch-guarded CAS rather than a full-row rewrite.
* Reply subject resolution is deterministic and fail-closed: exact referenced
  row, then a direct target id, then a UNIQUE same-audience label — never a
  guess, and never the requester.
* The quoted body is byte-absent from requester content, hash, and tags.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.proxy._envelope import _extract_envelope_metadata
from virtual_context.proxy.server import _roles_for_active_user
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnRow, Message

GUILD_KEY = "sk:agent:bast:discord:channel:15249"
OPTICS = "1111111111111111111"
BIGTEX = "2222222222222222222"
OPTICS_ACTOR = f"actor:discord:{OPTICS}"
BIGTEX_ACTOR = f"actor:discord:{BIGTEX}"


def _reconciler(store) -> IngestReconciler:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _store(tmp_path: Path, name: str = "vc.db") -> SQLiteStore:
    store = SQLiteStore(tmp_path / name)
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _anthropic():
    from virtual_context.proxy.formats import AnthropicFormat
    return AnthropicFormat()


def _openai():
    from virtual_context.proxy.formats import OpenAIFormat
    return OpenAIFormat()


def _conv_info(sender_id: str, **extra) -> str:
    import json
    payload = {"sender_id": sender_id, "chat_id": "channel:15249", **extra}
    return (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )


def _reply_block(**payload) -> str:
    import json
    return (
        "\n\nReply target of current user message (context):\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )


def _body(*texts: str) -> dict:
    """An Anthropic body whose user turns carry *texts*, assistant replies between."""
    messages = []
    for i, text in enumerate(texts):
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": f"reply {i}"})
    return {"messages": messages}


def _rows(store):
    return store.get_all_canonical_turns("c")


# ---------------------------------------------------------------------------
# W1 — actor id on the user row only
# ---------------------------------------------------------------------------

def test_ingest_batch_populates_actor_on_user_rows(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(_conv_info(BIGTEX) + "always rebase before merge"),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    rows = _rows(store)
    user = [r for r in rows if r.user_content]
    assistant = [r for r in rows if r.assistant_content and not r.user_content]

    assert user and user[0].sender_actor_id == BIGTEX_ACTOR
    # An assistant row is never newly labeled with a human actor.
    assert all(r.sender_actor_id == "" for r in assistant)

    profile = store._get_conn().execute(
        "SELECT tenant_id, actor_id FROM actor_profiles WHERE actor_id = ?",
        (BIGTEX_ACTOR,),
    ).fetchone()
    assert tuple(profile) == ("t", BIGTEX_ACTOR)


def test_actor_platform_comes_from_the_raw_key_not_the_resolved_id(tmp_path):
    """After a VCATTACH the engine id is a UUID that names no platform."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",  # resolved id — carries no platform
        body=_body(_conv_info(BIGTEX) + "hi"),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,  # raw caller key — carries it
    )
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == BIGTEX_ACTOR


def test_no_raw_key_and_a_uuid_conversation_yields_no_actor(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(_conv_info(BIGTEX) + "hi"),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
    )
    # Honest empty. Never actor:unknown:<id>.
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == ""


def test_openai_format_populates_actor_too(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body={"messages": [
            {"role": "user", "content": _conv_info(OPTICS) + "hello"},
            {"role": "assistant", "content": "hi"},
        ]},
        fmt=_openai(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == OPTICS_ACTOR


def test_empty_metadata_stays_empty(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c", body=_body("plain message"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    assert all(r.sender_actor_id == "" for r in _rows(store))


# ---------------------------------------------------------------------------
# W2 — preservation and the epoch-guarded CAS
# ---------------------------------------------------------------------------

def test_resend_without_envelope_cannot_blank_a_stored_actor(tmp_path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    rec.ingest_batch(
        "c", body=_body(_conv_info(BIGTEX) + "hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    # The same turn resent with the envelope gone — the hash is over stripped
    # content, so these align and preservation must hold.
    rec.ingest_batch(
        "c", body=_body("hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == BIGTEX_ACTOR


def test_overlap_fast_skip_upgrades_an_empty_actor_through_the_cas(tmp_path):
    """Aligned overlap rows are never rewritten, so the CAS is the only path."""
    store = _store(tmp_path)
    rec = _reconciler(store)
    rec.ingest_batch(
        "c", body=_body("hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == ""

    rec.ingest_batch(
        "c", body=_body(_conv_info(BIGTEX) + "hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == BIGTEX_ACTOR


def test_cas_never_overwrites_a_stored_actor(tmp_path):
    store = _store(tmp_path)
    _store_row = _reconciler(store)
    _store_row.ingest_batch(
        "c", body=_body(_conv_info(BIGTEX) + "hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    written = store.update_canonical_turn_actors_if_empty(
        "c", {row.canonical_turn_id: OPTICS_ACTOR},
    )
    assert written == 0
    assert [r for r in _rows(store) if r.user_content][0].sender_actor_id == BIGTEX_ACTOR


def test_cas_is_epoch_guarded(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c", body=_body("hello"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    stale = store.update_canonical_turn_actors_if_empty(
        "c", {row.canonical_turn_id: BIGTEX_ACTOR}, expected_lifecycle_epoch=999,
    )
    assert stale == 0

    ok = store.update_canonical_turn_actors_if_empty(
        "c", {row.canonical_turn_id: BIGTEX_ACTOR}, expected_lifecycle_epoch=1,
    )
    assert ok == 1


# ---------------------------------------------------------------------------
# W1a — the reply edge
# ---------------------------------------------------------------------------

def test_reply_edge_persists_on_the_user_row(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1")
            + "thoughts, Vast?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex",
                           body="always rebase before merge"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        # A native request whose route the resolver already proved. The
        # reconciler never infers this: an unproved route must not silently
        # become the owner.
        source_audience_conversation_id="c",
    )
    row = [r for r in _rows(store) if r.user_content][0]

    assert row.sender_actor_id == OPTICS_ACTOR        # requester
    assert row.reply_subject_actor_id == BIGTEX_ACTOR  # subject
    assert row.reply_subject_label == "BigTex"
    assert row.source_message_id == "m2"
    assert row.reply_target_message_id == "m1"
    assert row.reply_target_body == "always rebase before merge"
    assert row.reply_attribution_version == 1
    assert row.audience_conversation_id == "c"
    assert row.audience_attribution_version == 1


def test_an_unproved_route_leaves_the_row_policy_ineligible(tmp_path):
    """No proved audience means no audience — never a fallback to the owner.

    The owner scopes storage; the audience scopes disclosure. Defaulting an
    unproved route to the owner is exactly how a DM request routed through a
    retained VCMERGE alias would start receiving guild-origin influence.
    """
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(_conv_info(OPTICS, message_id="m2") + "hi"),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    # The actor still resolves: identity provenance is not an audience proof.
    assert row.sender_actor_id == OPTICS_ACTOR
    assert row.audience_conversation_id == ""
    assert row.audience_attribution_version == 0


def test_quoted_body_is_absent_from_requester_content_and_hash(tmp_path):
    """I11. The quoted claim must never look like the requester's own words."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2")
            + "thoughts, Vast?"
            + _reply_block(sender_label="BigTex", body="always rebase before merge"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    assert row.user_content == "thoughts, Vast?"
    assert "rebase" not in row.user_content
    assert "rebase" not in row.normalized_user_text
    # The body lives only in its own lane.
    assert row.reply_target_body == "always rebase before merge"


def test_assistant_row_never_receives_a_reply_edge(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1")
            + "thoughts?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex", body="x"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    for row in _rows(store):
        if row.assistant_content and not row.user_content:
            assert row.reply_subject_actor_id == ""
            assert row.reply_target_body == ""
            assert row.reply_attribution_version == 0


def test_subject_resolves_from_the_referenced_row_in_the_same_batch(tmp_path):
    """BigTex states a claim; Optics replies to that exact message."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(BIGTEX, message_id="m1") + "always rebase before merge",
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1")
            + "thoughts, Vast?"
            + _reply_block(sender_label="BigTex", body="always rebase before merge"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    users = [r for r in _rows(store) if r.user_content]
    bigtex_row = next(r for r in users if r.source_message_id == "m1")
    optics_row = next(r for r in users if r.source_message_id == "m2")

    assert bigtex_row.sender_actor_id == BIGTEX_ACTOR
    assert optics_row.sender_actor_id == OPTICS_ACTOR
    # Resolved from the referenced row, with no id in the reply block at all.
    assert optics_row.reply_subject_actor_id == BIGTEX_ACTOR
    assert optics_row.reply_subject_actor_id != optics_row.sender_actor_id


def test_subject_never_falls_back_to_the_requester(tmp_path):
    """The worst-failure fence: an unresolved subject stays empty."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2", reply_to_id="does-not-exist")
            + "thoughts?"
            + _reply_block(sender_label="Nobody", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    assert row.sender_actor_id == OPTICS_ACTOR
    assert row.reply_subject_actor_id == ""
    # The body survives as untrusted quoted context, still unattributed.
    assert row.reply_target_body == "a claim"


def test_ambiguous_label_leaves_the_subject_unresolved(tmp_path):
    """Two members share a display name: picking one would misattribute."""
    store = _store(tmp_path)
    rec = _reconciler(store)
    import json

    def _named(sender_id, name, mid, text):
        return (
            "Sender (member):\n```json\n" + json.dumps({"name": name}) + "\n```\n"
            + _conv_info(sender_id, message_id=mid)
            + text
        )

    rec.ingest_batch(
        "c",
        body=_body(
            _named(BIGTEX, "Tex", "m1", "first claim"),
            _named(OPTICS, "Tex", "m2", "second claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    # Both durable actors now answer to the label "Tex".
    assert store.find_actor_ids_by_display_label("c", "Tex") == sorted(
        [BIGTEX_ACTOR, OPTICS_ACTOR]
    )

    rec.ingest_batch(
        "c",
        body=_body(
            _named(BIGTEX, "Tex", "m1", "first claim"),
            _named(OPTICS, "Tex", "m2", "second claim"),
            _conv_info(OPTICS, message_id="m3") + "thoughts?"
            + _reply_block(sender_label="Tex", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = next(r for r in _rows(store) if r.source_message_id == "m3")
    assert row.reply_subject_actor_id == ""


def test_unique_label_resolves_the_subject(tmp_path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    import json

    first = (
        "Sender (member):\n```json\n" + json.dumps({"name": "BigTex"}) + "\n```\n"
        + _conv_info(BIGTEX, message_id="m1")
        + "always rebase"
    )
    rec.ingest_batch(
        "c", body=_body(first), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    rec.ingest_batch(
        "c",
        body=_body(
            first,
            _conv_info(OPTICS, message_id="m9") + "thoughts?"
            + _reply_block(sender_label="BigTex", body="always rebase"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    row = next(r for r in _rows(store) if r.source_message_id == "m9")
    assert row.reply_subject_actor_id == BIGTEX_ACTOR


def test_contradictory_row_and_block_ids_fail_closed(tmp_path):
    """Never pick by precedence across two contradictory actors."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(BIGTEX, message_id="m1") + "a claim",
            # The referenced row is BigTex's, but the block claims Optics.
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1") + "thoughts?"
            + _reply_block(sender_id=OPTICS, sender_label="Optics", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    row = next(r for r in _rows(store) if r.source_message_id == "m2")
    assert row.reply_subject_actor_id == ""


def test_same_batch_duplicate_message_ids_fail_closed(tmp_path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    prepared = [
        CanonicalTurnRow(
            conversation_id="c", canonical_turn_id="a", user_content="one",
            source_message_id="m1", sender_actor_id=BIGTEX_ACTOR,
            audience_conversation_id="c",
        ),
        CanonicalTurnRow(
            conversation_id="c", canonical_turn_id="b", user_content="two",
            source_message_id="m1", sender_actor_id=OPTICS_ACTOR,
            audience_conversation_id="c",
        ),
        CanonicalTurnRow(
            conversation_id="c", canonical_turn_id="reply", user_content="thoughts?",
            reply_target_message_id="m1", reply_attribution_version=1,
            audience_conversation_id="c",
        ),
    ]

    rec._resolve_reply_subjects("c", prepared)

    assert prepared[-1].reply_subject_actor_id == ""


def test_full_row_rewrite_preserves_a_conflicting_stored_actor_and_edge():
    stored = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="ct1", sender_actor_id=BIGTEX_ACTOR,
        source_message_id="m1", reply_target_message_id="m0",
        reply_subject_actor_id=BIGTEX_ACTOR,
        reply_target_body="original claim", audience_conversation_id="dm",
        reply_attribution_version=1, audience_attribution_version=1,
    )
    incoming = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="ct1", sender_actor_id=OPTICS_ACTOR,
        source_message_id="m2", reply_target_message_id="other",
        reply_subject_actor_id=OPTICS_ACTOR,
        reply_target_body="different claim", audience_conversation_id="guild",
        reply_attribution_version=1, audience_attribution_version=1,
    )

    IngestReconciler._preserve_existing_enrichment(incoming, stored)

    assert incoming.sender_actor_id == BIGTEX_ACTOR
    assert incoming.source_message_id == "m1"
    assert incoming.reply_target_message_id == "m0"
    assert incoming.reply_subject_actor_id == BIGTEX_ACTOR
    assert incoming.reply_target_body == "original claim"
    assert incoming.audience_conversation_id == "dm"


def test_full_row_conflict_does_not_fill_gaps_from_the_other_edge():
    stored = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="ct1",
        reply_subject_actor_id=BIGTEX_ACTOR,
    )
    incoming = CanonicalTurnRow(
        conversation_id="c", canonical_turn_id="ct1",
        reply_subject_actor_id=OPTICS_ACTOR,
        reply_target_body="Optics's claim", reply_attribution_version=1,
    )

    IngestReconciler._preserve_existing_enrichment(incoming, stored)

    assert incoming.reply_subject_actor_id == BIGTEX_ACTOR
    assert incoming.reply_target_body == ""
    assert incoming.reply_attribution_version == 0


def test_tagger_full_row_rewrites_resupply_audience_origin():
    row = CanonicalTurnRow(
        conversation_id="c",
        audience_conversation_id="dm", audience_attribution_version=1,
    )

    edge = TaggingPipeline._row_reply_edge(row)

    assert edge["audience_conversation_id"] == "dm"
    assert edge["audience_attribution_version"] == 1


def test_proxy_roles_come_from_the_active_entry_and_validated_audience(tmp_path):
    store = SQLiteStore(tmp_path / "roles.db")
    store.upsert_conversation(tenant_id="t1", conversation_id=GUILD_KEY)
    store.save_canonical_turn(
        GUILD_KEY, 0, "claim", "", canonical_turn_id="target",
        turn_hash="target-hash", sort_key=1.0, sender="BigTex",
        sender_actor_id=BIGTEX_ACTOR, source_message_id="m1",
        audience_conversation_id=GUILD_KEY, audience_attribution_version=1,
        origin_channel_id="15249",
    )
    raw = (
        _conv_info(OPTICS, message_id="m2", reply_to_id="m1")
        + "thoughts?"
        + _reply_block(sender_label="BigTex", body="claim")
    )
    text, metadata = _extract_envelope_metadata(raw)
    active = Message(role="user", content=text, metadata=metadata)
    engine = SimpleNamespace(
        config=SimpleNamespace(conversation_id=GUILD_KEY, tenant_id="t1"),
        _store=store,
    )
    engine._ingest_reconciler = _reconciler(store)
    state = SimpleNamespace(engine=engine)

    roles = _roles_for_active_user(
        state, active, "thoughts?",
        inbound_conversation_id=GUILD_KEY,
        audience_conversation_id=GUILD_KEY,
    )

    assert roles.requester_actor_id == OPTICS_ACTOR
    assert roles.subject_actor_id == BIGTEX_ACTOR
    assert roles.reply_target_message_id == "m1"
    assert roles.audience_conversation_id == GUILD_KEY
    assert roles.audience_channel_id == "15249"


def test_proxy_role_selection_mismatch_fails_closed(tmp_path):
    store = SQLiteStore(tmp_path / "roles-mismatch.db")
    store.upsert_conversation(tenant_id="t1", conversation_id="owner")
    state = SimpleNamespace(engine=SimpleNamespace(
        config=SimpleNamespace(conversation_id="owner", tenant_id="t1"),
        _store=store,
    ))

    roles = _roles_for_active_user(
        state, Message(role="user", content="other", metadata={}), "active",
        inbound_conversation_id="owner", audience_conversation_id="owner",
    )

    assert roles.requester_actor_id == ""
    assert roles.subject_actor_id == ""
    assert roles.audience_conversation_id == ""


def test_reply_edge_preserved_across_a_resend_without_the_block(tmp_path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    rec.ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1") + "thoughts?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    rec.ingest_batch(
        "c", body=_body("thoughts?"), fmt=_anthropic(),
        expected_lifecycle_epoch=1, source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]
    assert row.reply_subject_actor_id == BIGTEX_ACTOR
    assert row.reply_target_body == "a claim"


def test_reply_cas_never_rewrites_a_contradictory_stored_edge(tmp_path):
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2") + "thoughts?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    store.update_canonical_turn_reply_roles_if_empty(
        "c",
        {row.canonical_turn_id: {
            "reply_subject_actor_id": OPTICS_ACTOR,
            "reply_target_body": "a different claim",
        }},
    )
    after = [r for r in _rows(store) if r.user_content][0]
    # The stored edge wins. Rewriting it would move a claim between members.
    assert after.reply_subject_actor_id == BIGTEX_ACTOR
    assert after.reply_target_body == "a claim"


def test_reply_cas_rejects_whole_partial_edge_on_one_conflict(tmp_path):
    store = _store(tmp_path)
    store.save_canonical_turn(
        "c", -1, "thoughts?", "", canonical_turn_id="partial",
        sort_key=1, turn_hash="partial", reply_subject_actor_id=BIGTEX_ACTOR,
    )

    updated = store.update_canonical_turn_reply_roles_if_empty(
        "c",
        {"partial": {
            "reply_subject_actor_id": OPTICS_ACTOR,
            "reply_target_body": "Optics's claim",
            "reply_attribution_version": 1,
        }},
    )
    row = store.get_all_canonical_turns("c")[0]

    assert updated == 0
    assert row.reply_subject_actor_id == BIGTEX_ACTOR
    assert row.reply_target_body == ""
    assert row.reply_attribution_version == 0


def test_vcmerge_may_move_a_duplicate_source_id_and_lookup_fails_closed(tmp_path):
    """The (conversation_id, source_message_id) index must NOT be unique.

    Platform message ids are opaque, and VCMERGE moves the source and target
    conversations' canonical rows under ONE owner. Two rows can therefore
    legitimately claim the same id under that owner, and a unique index would
    reject an otherwise valid merge. Ambiguity is not prevented at write time;
    it is resolved at lookup, where it fails closed.
    """
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c", body=_body(_conv_info(BIGTEX, message_id="m1") + "hi"),
        fmt=_anthropic(), expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
        source_audience_conversation_id="c",
    )
    conn = store._get_conn()
    # The merged-in row: same owner, same opaque message id, but it was
    # observed on a DIFFERENT route (the retained source alias).
    conn.execute(
        "INSERT INTO canonical_turns "
        "(canonical_turn_id, conversation_id, sort_key, turn_hash, "
        " user_content, source_message_id, sender_actor_id, "
        " audience_conversation_id, audience_attribution_version) "
        "VALUES ('x', 'c', 99999.0, 'h', 'other', 'm1', ?, 'dm', 1)",
        (OPTICS_ACTOR,),
    )
    conn.commit()

    # Owner alone is not a boundary: both rows share it, so this is ambiguous
    # and must resolve to nothing rather than guessing which member was quoted.
    assert store.find_canonical_turn_by_source_message_id("c", "m1") is None

    # The validated pre-alias route disambiguates. Each audience sees only the
    # row actually observed on it.
    dm = store.find_canonical_turn_by_source_message_id(
        "c", "m1", audience_conversation_id="dm",
    )
    assert dm is not None and dm.sender_actor_id == OPTICS_ACTOR
    guild = store.find_canonical_turn_by_source_message_id(
        "c", "m1", audience_conversation_id="c",
    )
    assert guild is not None and guild.sender_actor_id == BIGTEX_ACTOR


# ---------------------------------------------------------------------------
# W2 — the tagger's full-row rewrites must not erase enrichment
# ---------------------------------------------------------------------------

def test_background_tagging_rewrite_preserves_the_actor_and_reply_edge(tmp_path):
    """A full-row upsert defaults every omitted column away.

    The tagger rewrites canonical rows in place. If it forgets to re-supply
    the actor or the reply edge, tagging silently erases attribution that
    ingest had already resolved correctly.
    """
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2", reply_to_id="m1") + "thoughts?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]
    assert row.sender_actor_id == OPTICS_ACTOR
    assert row.reply_subject_actor_id == BIGTEX_ACTOR

    # Simulate the tagger's in-place rewrite: same row, new tags, and every
    # enrichment column re-supplied from the stored row.
    store.save_canonical_turn(
        "c",
        -1,
        row.user_content,
        row.assistant_content,
        canonical_turn_id=row.canonical_turn_id,
        sort_key=row.sort_key,
        turn_hash=row.turn_hash,
        primary_tag="git-workflow",
        tags=["git-workflow"],
        sender=row.sender,
        origin_channel_id=row.origin_channel_id,
        origin_channel_label=row.origin_channel_label,
        sender_actor_id=row.sender_actor_id,
        source_message_id=row.source_message_id,
        reply_target_message_id=row.reply_target_message_id,
        reply_subject_actor_id=row.reply_subject_actor_id,
        reply_subject_label=row.reply_subject_label,
        reply_target_body=row.reply_target_body,
        reply_attribution_version=row.reply_attribution_version,
        audience_attribution_version=row.audience_attribution_version,
    )
    after = [r for r in _rows(store) if r.user_content][0]
    assert after.primary_tag == "git-workflow"
    assert after.sender_actor_id == OPTICS_ACTOR
    assert after.reply_subject_actor_id == BIGTEX_ACTOR
    assert after.reply_target_body == "a claim"
    assert after.source_message_id == "m2"


def test_tagger_rewrite_that_omits_the_edge_would_erase_it(tmp_path):
    """Pins WHY every tagger call site must re-supply the edge explicitly."""
    store = _store(tmp_path)
    _reconciler(store).ingest_batch(
        "c",
        body=_body(
            _conv_info(OPTICS, message_id="m2") + "thoughts?"
            + _reply_block(sender_id=BIGTEX, sender_label="BigTex", body="a claim"),
        ),
        fmt=_anthropic(),
        expected_lifecycle_epoch=1,
        source_conversation_key=GUILD_KEY,
    )
    row = [r for r in _rows(store) if r.user_content][0]

    # A rewrite that omits the enrichment kwargs.
    store.save_canonical_turn(
        "c", -1, row.user_content, row.assistant_content,
        canonical_turn_id=row.canonical_turn_id,
        sort_key=row.sort_key, turn_hash=row.turn_hash,
    )
    after = [r for r in _rows(store) if r.user_content][0]
    # This is the failure mode the call sites must prevent.
    assert after.sender_actor_id == ""
    assert after.reply_subject_actor_id == ""
