"""Speaker roster in assembly: gating, budgeting, hard-cap rebuild, schema.

The gate ships dark: off means zero roster or handle-assignment reads, no
budget key, and rendered output plus tool schemas byte-identical to before
the feature existed. On, the roster is charged wrapper-inclusive exactly
once, participates in the final hard-cap rebuild by whole-entry eviction,
and the surviving snapshot is the single source for both the rendered block
and any request-local ``speaker`` enum — they can never disagree. Execution
does not read a ``speaker`` argument yet: an arriving value changes nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.retrieval_assembler import RetrievalAssembler
from virtual_context.core.speaker_roster import render_speaker_roster
from virtual_context.core.tool_loop import (
    execute_vc_tool,
    vc_tool_definitions,
    vc_tool_definitions_for_runtime,
)
from virtual_context.core.tool_guard import reset_default_guard
from virtual_context.types import (
    AssembledContext,
    AssemblerConfig,
    CanonicalTurnRow,
    EngineState,
    Message,
    RetrievalResult,
    RequestRoles,
    RetrieverConfig,
    SearchConfig,
    SpeakerHandleAssignment,
    SpeakerRetrievalContext,
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
    TagGeneratorConfig,
    VirtualContextConfig,
)

OWNER = "conv-1"
GUILD = "conv-1"
DM_ALIAS = "conv-dm"

ALEX = "actor:discord:alex"
BEA = "actor:discord:bea"
SANIA = "actor:telegram:sania"


def _tc(text: str) -> int:
    return len(text)


def _row(actor, sort_key, *, audience=GUILD, sender=None, channel=""):
    if sender is None:
        sender = actor.rsplit(":", 1)[-1].title()
    return CanonicalTurnRow(
        conversation_id=OWNER,
        canonical_turn_id=f"ct-{actor}-{sort_key}",
        sort_key=float(sort_key),
        user_content="hello",
        sender=sender,
        sender_actor_id=actor,
        audience_conversation_id=audience,
        audience_attribution_version=1,
        origin_channel_id=channel,
    )


class RosterStore:
    """Spy store with physical rows, durable handles, and an actor card."""

    def __init__(self, rows=None, assigned=None, cards=None):
        self.rows = list(rows or [])
        self.assigned = dict(assigned or {})
        self.cards = dict(cards or {})
        self.scan_calls: list = []
        self.fetch_calls: list = []
        self.alloc_calls: list = []
        self.card_calls: list = []

    def get_recent_canonical_turns(self, conversation_id, *, limit):
        self.scan_calls.append((conversation_id, limit))
        rows = [r for r in self.rows if r.conversation_id == conversation_id]
        rows.sort(key=lambda r: (r.sort_key, r.canonical_turn_id), reverse=True)
        return rows[:limit]

    def get_lifecycle_epoch(self, conversation_id):
        return 3

    def supports_speaker_handles(self):
        return True

    def _assignment(self, tenant_id, audience_conversation_id, actor):
        return SpeakerHandleAssignment(
            tenant_id=tenant_id,
            audience_conversation_id=audience_conversation_id,
            actor_id=actor,
            handle=self.assigned[(tenant_id, audience_conversation_id, actor)],
        )

    def get_speaker_handles(self, tenant_id, audience_conversation_id, actor_ids):
        self.fetch_calls.append(
            (tenant_id, audience_conversation_id, list(actor_ids)),
        )
        return [
            self._assignment(tenant_id, audience_conversation_id, actor)
            for actor in actor_ids
            if (tenant_id, audience_conversation_id, actor) in self.assigned
        ]

    def allocate_speaker_handles(self, tenant_id, audience_conversation_id,
                                 candidates, *, owner_conversation_id,
                                 expected_lifecycle_epoch):
        self.alloc_calls.append(list(candidates))
        taken = {
            handle for (t, aud, _), handle in self.assigned.items()
            if t == tenant_id and aud == audience_conversation_id
        }
        out = []
        for cand in candidates:
            key = (tenant_id, audience_conversation_id, cand.actor_id)
            if key not in self.assigned:
                base = cand.normalized_base or "user"
                handle = base
                suffix = 2
                while handle in taken:
                    handle = f"{base}.{suffix}"
                    suffix += 1
                taken.add(handle)
                self.assigned[key] = handle
            out.append(
                self._assignment(
                    tenant_id, audience_conversation_id, cand.actor_id,
                ),
            )
        return out

    def get_actor_card(self, tenant_id, actor_id, *, owner_conversation_id,
                       audience_conversation_id, audience_channel_id=""):
        self.card_calls.append((tenant_id, actor_id))
        return self.cards.get((tenant_id, actor_id))


def _assembler(store, *, roster_enabled=True, roster_max_tokens=2000,
               card_enabled=False, tenant="t1"):
    config = AssemblerConfig(
        speaker_roster_enabled=roster_enabled,
        speaker_roster_max_tokens=roster_max_tokens,
        actor_card_enabled=card_enabled,
    )
    return ContextAssembler(
        config=config,
        token_counter=_tc,
        store=store,
        conversation_id=OWNER,
        tenant_id=tenant,
    )


def _ctx(audience=GUILD, channel="", tenant="t1"):
    return SpeakerRetrievalContext(
        tenant_id=tenant,
        owner_conversation_id=OWNER,
        audience_conversation_id=audience,
        audience_channel_id=channel,
        requester_actor_id=ALEX,
    )


def _assemble(asm, ctx=None, *, budget=100_000, roles=None):
    return asm.assemble(
        core_context="CORE",
        retrieval_result=RetrievalResult(summaries=[], facts=[]),
        conversation_history=[Message(role="user", content="hi")],
        token_budget=budget,
        context_hint="HINT",
        request_roles=roles,
        speaker_context=ctx,
    )


def _three_member_store(**kw):
    return RosterStore(rows=[
        _row(ALEX, 30.0, sender="Alex"),
        _row(BEA, 20.0, sender="Bea"),
        _row(SANIA, 10.0, sender="Sania"),
    ], **kw)


def _roster_handles(text: str) -> list[str]:
    payload = json.loads(text.splitlines()[1])
    return [s["handle"] for s in payload["speakers"]]


# ---------------------------------------------------------------------------
# Ship dark
# ---------------------------------------------------------------------------

def test_gate_off_reads_nothing_renders_nothing_adds_no_budget_key():
    store = _three_member_store()
    out = _assemble(_assembler(store, roster_enabled=False), _ctx())

    assert out.speaker_roster_text == ""
    assert out.speaker_roster_snapshot is None
    assert out.speaker_context is None
    assert "speaker-roster" not in out.prepend_text
    assert "speaker_roster" not in out.budget_breakdown
    # No membership scan, no assignment read, no allocation: nothing at all.
    assert store.scan_calls == []
    assert store.fetch_calls == []
    assert store.alloc_calls == []


def test_gate_off_prepend_matches_gate_on_with_no_members():
    empty = RosterStore()
    off = _assemble(_assembler(empty, roster_enabled=False), _ctx())
    on = _assemble(_assembler(RosterStore(), roster_enabled=True), _ctx())
    assert off.prepend_text == on.prepend_text


def test_ineligible_context_builds_nothing_even_with_gate_on():
    store = _three_member_store()
    out = _assemble(_assembler(store), _ctx(audience=""))
    assert out.speaker_roster_text == ""
    assert store.scan_calls == []

    out2 = _assemble(_assembler(_three_member_store()), None)
    assert out2.speaker_roster_text == ""


# ---------------------------------------------------------------------------
# Placement and charging
# ---------------------------------------------------------------------------

def test_roster_is_injected_after_core_and_before_hint():
    out = _assemble(_assembler(_three_member_store()), _ctx())
    assert out.speaker_roster_text
    assert out.speaker_roster_text in out.prepend_text
    assert out.prepend_text.index("CORE") \
        < out.prepend_text.index("<speaker-roster") \
        < out.prepend_text.index("HINT")


def test_roster_charged_exactly_once_and_totals_add_up():
    out = _assemble(_assembler(_three_member_store()), _ctx())
    bd = out.budget_breakdown
    assert bd["speaker_roster"] == _tc(out.speaker_roster_text)
    assert out.total_tokens == (
        bd["core"] + bd["context_hint"] + bd["tags"] + bd["facts"]
        + bd["speaker_roster"] + bd["conversation"]
    )


def test_wrapper_inclusive_cap_drops_whole_least_recent_entries():
    generous = _assemble(_assembler(_three_member_store()), _ctx())
    full_tokens = generous.budget_breakdown["speaker_roster"]

    capped_asm = _assembler(
        _three_member_store(), roster_max_tokens=full_tokens - 1,
    )
    out = _assemble(capped_asm, _ctx())

    snap = out.speaker_roster_snapshot
    assert snap is not None
    assert 0 < len(snap.entries) < 3
    assert snap.truncated is True
    assert out.budget_breakdown["speaker_roster"] <= full_tokens - 1
    # Most recent survive: Sania (least recent) is the first to go.
    assert [e.actor_id for e in snap.entries] == [ALEX, BEA][:len(snap.entries)]
    # Surviving scalars are intact, never cut mid-name.
    payload = json.loads(out.speaker_roster_text.splitlines()[1])
    assert {s["name"] for s in payload["speakers"]} <= {"Alex", "Bea", "Sania"}


# ---------------------------------------------------------------------------
# Hard-cap rebuild
# ---------------------------------------------------------------------------

def test_hard_cap_rebuild_keeps_rendered_roster_snapshot_and_enum_in_sync():
    store = _three_member_store()
    generous = _assemble(_assembler(store), _ctx())
    full_prepend = _tc(generous.prepend_text)
    roster_tokens = generous.budget_breakdown["speaker_roster"]

    # A budget that forces the rebuild to evict some (not all) roster
    # entries: there are no tag sections or card entries to shed first.
    tight = full_prepend - (roster_tokens // 3)
    out = _assemble(_assembler(_three_member_store()), _ctx(), budget=tight)

    snap = out.speaker_roster_snapshot
    assert snap is not None
    assert 0 < len(snap.entries) < 3
    # The rendered block IS the render of the surviving snapshot.
    assert out.speaker_roster_text == render_speaker_roster(snap)
    assert out.speaker_roster_text in out.prepend_text
    assert _tc(out.prepend_text) <= tight
    # The enum built from the surviving snapshot matches the visible roster.
    defs = vc_tool_definitions_for_runtime(
        None, restore_available=False, roster_snapshot=snap,
    )
    find_quote = next(d for d in defs if d["name"] == "vc_find_quote")
    enum = find_quote["input_schema"]["properties"]["speaker"]["enum"]
    assert enum == _roster_handles(out.speaker_roster_text)
    # The snapshot id never changed across eviction: the request context
    # still points at the surviving snapshot.
    assert out.speaker_context is not None
    assert out.speaker_context.roster_snapshot_id == snap.snapshot_id
    # And the charge tracks the surviving render.
    assert out.budget_breakdown["speaker_roster"] == _tc(out.speaker_roster_text)


def test_full_eviction_emits_no_roster_no_context_and_no_speaker_parameter():
    store = _three_member_store()
    baseline = _assemble(_assembler(RosterStore()), _ctx())
    floor = _tc(baseline.prepend_text)

    # Budget below even the roster-free prepend: every entry and then the
    # wrapper itself must go.
    out = _assemble(_assembler(store), _ctx(), budget=floor)

    assert out.speaker_roster_text == ""
    assert out.speaker_roster_snapshot is None
    assert out.speaker_context is None
    assert "speaker-roster" not in out.prepend_text
    defs = vc_tool_definitions_for_runtime(
        None, restore_available=False,
        roster_snapshot=out.speaker_roster_snapshot,
    )
    assert all(
        "speaker" not in d["input_schema"].get("properties", {}) for d in defs
    )


def test_snapshot_id_is_stable_within_the_request():
    out = _assemble(_assembler(_three_member_store()), _ctx())
    assert out.speaker_roster_snapshot is not None
    assert out.speaker_context.roster_snapshot_id \
        == out.speaker_roster_snapshot.snapshot_id != ""


# ---------------------------------------------------------------------------
# Gate independence
# ---------------------------------------------------------------------------

def test_actor_card_and_roster_gates_are_independent():
    from virtual_context.types import (
        CARD_KIND_COMMUNICATION_PREF,
        ActorCard,
        ActorCardEntry,
    )

    def _store():
        return _three_member_store(cards={("t1", ALEX): ActorCard(
            tenant_id="t1", actor_id=ALEX, display_name="Alex",
            entries=[ActorCardEntry(
                id="e1", kind=CARD_KIND_COMMUNICATION_PREF,
                body="prefers terse answers", confidence=0.9,
                sensitivity="normal", updated_at="2026-01-01",
            )],
        )})

    roles = RequestRoles(
        requester_actor_id=ALEX,
        owner_conversation_id=OWNER,
        audience_conversation_id=GUILD,
    )

    # Card on, roster off: card renders, zero roster operations.
    store = _store()
    out = _assemble(
        _assembler(store, roster_enabled=False, card_enabled=True),
        _ctx(), roles=roles,
    )
    assert "<actor-card" in out.prepend_text
    assert "speaker-roster" not in out.prepend_text
    assert store.scan_calls == [] and store.fetch_calls == []
    assert "actor_card" in out.budget_breakdown
    assert "speaker_roster" not in out.budget_breakdown

    # Roster on, card off: roster renders, zero card reads.
    store = _store()
    out = _assemble(
        _assembler(store, roster_enabled=True, card_enabled=False),
        _ctx(), roles=roles,
    )
    assert "<speaker-roster" in out.prepend_text
    assert "actor-card" not in out.prepend_text
    assert store.card_calls == []
    assert "speaker_roster" in out.budget_breakdown
    assert "actor_card" not in out.budget_breakdown


# ---------------------------------------------------------------------------
# Dynamic schema surface
# ---------------------------------------------------------------------------

def test_schema_without_snapshot_is_byte_identical():
    plain = vc_tool_definitions_for_runtime(None, restore_available=False)
    explicit_none = vc_tool_definitions_for_runtime(
        None, restore_available=False, roster_snapshot=None,
    )
    legacy = [
        d for d in vc_tool_definitions() if d["name"] != "vc_restore_tool"
    ]
    assert json.dumps(plain, sort_keys=True) \
        == json.dumps(explicit_none, sort_keys=True) \
        == json.dumps(legacy, sort_keys=True)


def test_empty_snapshot_leaves_schema_byte_identical():
    empty = SpeakerRosterSnapshot(snapshot_id="s-empty")
    with_empty = vc_tool_definitions_for_runtime(
        None, restore_available=False, roster_snapshot=empty,
    )
    plain = vc_tool_definitions_for_runtime(None, restore_available=False)
    assert json.dumps(with_empty, sort_keys=True) \
        == json.dumps(plain, sort_keys=True)


def test_schema_with_snapshot_binds_handles_only():
    snap = SpeakerRosterSnapshot(
        snapshot_id="s1",
        entries=(
            SpeakerRosterEntry(handle="alex", name="Alex Q", actor_id=ALEX),
            SpeakerRosterEntry(handle="bea", name="Bea<script>",
                               actor_id=BEA),
        ),
    )
    defs = vc_tool_definitions_for_runtime(
        None, restore_available=True, roster_snapshot=snap,
    )
    rendered = json.dumps(defs)

    selectable = {"vc_find_quote", "vc_query_facts", "vc_remember_when"}
    for definition in defs:
        schema = definition["input_schema"]
        properties = schema.get("properties", {})
        if definition["name"] in selectable:
            speaker = properties["speaker"]
            assert speaker["enum"] == ["alex", "bea"]
            assert speaker["type"] == "string"
            # Selection is optional: never added to required.
            assert "speaker" not in schema.get("required", [])
        else:
            assert "speaker" not in properties

    # Handles only: no display names, no actor ids, anywhere in the schema.
    assert "Alex Q" not in rendered
    assert "Bea<script>" not in rendered
    assert ALEX not in rendered
    assert BEA not in rendered
    assert "actor:" not in rendered


# ---------------------------------------------------------------------------
# Assembly seam: the context flows through RetrievalAssembler untouched
# ---------------------------------------------------------------------------

def test_retrieval_assembler_forwards_speaker_context_to_assembly():
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        tags_matched=["topic"], summaries=[], facts=[],
        retrieval_metadata={"tags_from_message": ["topic"]},
    )
    assembler_delegate = MagicMock()
    assembler_delegate.assemble.return_value = AssembledContext()
    monitor = MagicMock()
    monitor.build_snapshot.return_value = SimpleNamespace(
        total_tokens=0, budget_tokens=10_000,
    )
    paging = MagicMock()
    paging.working_set = {}
    turn_tag_index = MagicMock()
    turn_tag_index.entries = []

    ra = RetrievalAssembler(
        retriever=retriever,
        assembler=assembler_delegate,
        monitor=monitor,
        paging=paging,
        store=MagicMock(),
        turn_tag_index=turn_tag_index,
        engine_state=EngineState(),
        fact_curator=None,
        config=VirtualContextConfig(
            context_window=10_000,
            tag_generator=TagGeneratorConfig(type="keyword"),
            retriever=RetrieverConfig(),
            assembler=AssemblerConfig(),
        ),
        token_counter=lambda text: len(text) // 4,
        session_state_provider=None,
    )
    ra._set_semantic(MagicMock())

    ctx = _ctx()
    ra.on_message_inbound(
        "what did bea say", [], request_roles=None, speaker_context=ctx,
    )
    kwargs = assembler_delegate.assemble.call_args.kwargs
    assert kwargs["speaker_context"] is ctx


# ---------------------------------------------------------------------------
# Execution does not read a speaker argument
# ---------------------------------------------------------------------------

def test_execution_ignores_an_arriving_speaker_argument():
    reset_default_guard()

    def _engine():
        engine = MagicMock()
        engine.config.search = SearchConfig(tool_guard_enabled=False)
        engine.config.conversation_id = "conv-exec"
        engine.find_quote.return_value = {
            "found": True,
            "results": [{"excerpt": "the dosage was 5mg", "turn": 3}],
        }
        engine._store.search_facts.return_value = []
        return engine

    plain_engine = _engine()
    plain = execute_vc_tool(
        plain_engine, "vc_find_quote", {"query": "dosage", "mode": "lookup"},
    )
    speaker_engine = _engine()
    with_speaker = execute_vc_tool(
        speaker_engine, "vc_find_quote",
        {"query": "dosage", "mode": "lookup", "speaker": "alex"},
    )

    assert json.loads(plain) == json.loads(with_speaker)
    # The argument was never consumed: the search entrypoint received the
    # exact same call either way.
    assert plain_engine.find_quote.call_args == speaker_engine.find_quote.call_args
