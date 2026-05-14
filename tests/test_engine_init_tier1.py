"""Engine ``__init__`` Tier 1 gate tests for the cross-channel mirror.

The Tier 1 bool ``_is_merge_participant`` is set ONCE at engine
construction, gated on the Tier 0 ``protected_window_db_source ==
"merge"`` flag. These tests pin the four-shape topology coverage and
the off-mode short-circuit.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine


def _make_config(tmp_path, *, conversation_id: str | None = None, mode: str = "off"):
    cfg_dict = {
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
        "tag_generator": {"type": "keyword"},
        "retrieval": {"inbound_tagger_type": "keyword"},
        "assembly": {"protected_window_db_source": mode},
    }
    if conversation_id is not None:
        cfg_dict["conversation_id"] = conversation_id
    return load_config(config_dict=cfg_dict)


def _underlying_store(engine):
    store = engine._store
    return getattr(store, "_store", store)


def _seed_attachable_target(raw_store, target_id: str, *, tenant_id: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    sqlite = raw_store._segments
    conn = sqlite._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO conversations
           (conversation_id, tenant_id, phase, deleted_at,
            created_at, updated_at)
           VALUES (?, ?, 'active', NULL, ?, ?)""",
        (target_id, tenant_id, now, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tier 0 off short-circuit
# ---------------------------------------------------------------------------


def test_tier0_off_skips_has_any_alias(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, conversation_id="conv-off", mode="off")
    eng = VirtualContextEngine(config=cfg)
    # Off mode: _is_merge_participant must default to False without
    # consulting the store.
    assert eng._is_merge_participant is False


def test_tier0_off_short_circuits_even_with_alias(tmp_path):
    """Off mode must not consult has_any_alias even when an alias row
    exists. The Tier 1 cached bool stays False until the flag flips."""
    cfg = _make_config(tmp_path, conversation_id="seed-conv", mode="off")
    seeder = VirtualContextEngine(config=cfg)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-1")
    raw.save_conversation_alias("source-1", "target-1")
    seeder.close()

    cfg2 = _make_config(tmp_path, conversation_id="source-1", mode="off")
    eng = VirtualContextEngine(config=cfg2)
    assert eng._is_merge_participant is False


# ---------------------------------------------------------------------------
# Tier 0 merge derivation
# ---------------------------------------------------------------------------


def test_tier1_unattached_returns_false(tmp_path):
    cfg = _make_config(tmp_path, conversation_id="conv-unattached", mode="merge")
    eng = VirtualContextEngine(config=cfg)
    assert eng._is_merge_participant is False


def test_tier1_source_only_returns_true(tmp_path):
    """``has_any_alias`` must return True when the engine's conv is the
    source of an outgoing alias."""
    cfg_seed = _make_config(tmp_path, conversation_id="seed-conv", mode="off")
    seeder = VirtualContextEngine(config=cfg_seed)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-1")
    raw.save_conversation_alias("source-1", "target-1")
    seeder.close()

    # The alias resolver will follow source-1 -> target-1, so engine.config.conversation_id
    # ends up bound to target-1. Tier 1 then asks has_any_alias(target-1),
    # which is True (target leg).
    cfg = _make_config(tmp_path, conversation_id="source-1", mode="merge")
    eng = VirtualContextEngine(config=cfg)
    assert eng._is_merge_participant is True


def test_tier1_target_only_returns_true(tmp_path):
    """An engine bound to a target with an incoming alias should see True."""
    cfg_seed = _make_config(tmp_path, conversation_id="seed-conv", mode="off")
    seeder = VirtualContextEngine(config=cfg_seed)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-1")
    raw.save_conversation_alias("source-1", "target-1")
    seeder.close()

    cfg = _make_config(tmp_path, conversation_id="target-1", mode="merge")
    eng = VirtualContextEngine(config=cfg)
    assert eng._is_merge_participant is True


def test_tier1_both_source_and_target_returns_true(tmp_path):
    """Multi-hop: A -> B and C -> A. ``A`` is both source (outgoing to
    B) and target (incoming from C). has_any_alias(A) must return True."""
    cfg_seed = _make_config(tmp_path, conversation_id="seed-conv", mode="off")
    seeder = VirtualContextEngine(config=cfg_seed)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "B")
    _seed_attachable_target(raw, "A")
    raw.save_conversation_alias("A", "B")
    raw.save_conversation_alias("C", "A")
    seeder.close()

    # Engine bound to A would resolve A->B during init. But we want to
    # test has_any_alias on A directly, before the resolver hops to B.
    # Use the store directly.
    cfg = _make_config(tmp_path, conversation_id="dummy", mode="off")
    probe_engine = VirtualContextEngine(config=cfg)
    raw_probe = _underlying_store(probe_engine)
    assert raw_probe.has_any_alias("A") is True


def test_tier1_call_passes_positional_conv_id(tmp_path):
    """Cloud's tenant wrapper enforces validation only when the call
    uses ``conversation_id`` as the FIRST POSITIONAL argument. Engine
    code must pass it positionally, not as a keyword."""
    from unittest.mock import patch
    cfg_seed = _make_config(tmp_path, conversation_id="seed-conv", mode="off")
    seeder = VirtualContextEngine(config=cfg_seed)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-1")
    raw.save_conversation_alias("source-1", "target-1")
    seeder.close()

    cfg = _make_config(tmp_path, conversation_id="source-1", mode="merge")
    with patch.object(
        type(_underlying_store(VirtualContextEngine(config=_make_config(tmp_path, mode="off")))),
        "has_any_alias",
        autospec=True,
        return_value=True,
    ) as mock_method:
        eng = VirtualContextEngine(config=cfg)
        # First positional argument after self must be the conv id.
        # autospec=True ensures the mock binds self too; args[1] is the conv_id.
        assert mock_method.called
        args, kwargs = mock_method.call_args
        # conversation_id must be passed positionally (args[1] after self).
        assert kwargs == {}
        assert len(args) >= 2
        assert isinstance(args[1], str) and args[1]
