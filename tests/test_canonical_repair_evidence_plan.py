from __future__ import annotations

from scripts.apply_canonical_repair_artifact_20260802 import (
    OWNER,
    RepairError,
    _evidence_remap_plan,
    _invalidate_actor_cards,
)


def _snapshots(origin: str) -> dict[str, list[dict]]:
    return {
        "tool_outputs": [{
            "ref": "tool-1",
            "turn": 7,
            "origin_conversation_id": origin,
        }],
        "turn_tool_outputs": [{
            "conversation_id": OWNER,
            "turn_number": 7,
            "tool_output_ref": "tool-1",
            "origin_conversation_id": origin,
        }],
        "chain_snapshots": [],
    }


def test_foreign_origin_ordinal_is_quarantined_not_owner_mapped() -> None:
    plan = _evidence_remap_plan(
        snapshot_rows=_snapshots("sk:agent:vast:discord:channel:123"),
        old_group_map={7: 70},
    )
    assert plan["mapped_tools"] == []
    assert plan["mapped_links"] == []
    assert plan["quarantined"]["tool_outputs"][0]["reason"] == (
        "foreign_origin_ordinal_namespace"
    )
    assert plan["quarantined"]["turn_tool_outputs"]


def test_owner_origin_ordinal_maps_and_preserves_namespace() -> None:
    plan = _evidence_remap_plan(
        snapshot_rows=_snapshots(OWNER),
        old_group_map={7: 70},
    )
    assert [(row["ref"], target) for row, target in plan["mapped_tools"]] == [
        ("tool-1", 70),
    ]
    assert plan["mapped_links"] == [(OWNER, 70, "tool-1", OWNER)]
    assert not plan["quarantined"]["tool_outputs"]
    assert not plan["quarantined"]["turn_tool_outputs"]


def test_actor_card_invalidation_refuses_cross_tenant_sources() -> None:
    class _Result:
        @staticmethod
        def fetchone() -> dict[str, int]:
            return {"turns": 1, "facts": 0}

    class _Connection:
        @staticmethod
        def execute(_sql: str, _params: tuple[object, ...]) -> _Result:
            return _Result()

    try:
        _invalidate_actor_cards(
            _Connection(), OWNER, "men-tenant", set(),
        )
    except RepairError as exc:
        assert str(exc) == "cross-tenant actor-card evidence cites Men history"
    else:
        raise AssertionError("cross-tenant actor-card evidence was not refused")
