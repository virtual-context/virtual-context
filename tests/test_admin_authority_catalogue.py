"""Admin actor-backfill authority is absent from every model catalogue.

``backfill_actors`` and its ``--platform`` assertion are operator-only CLI
and engine-admin authority. No model-accessible tool surface — the VC tool
catalogue in any runtime shape, a request-local speaker schema, or the MCP
server's registered tools — may name, describe, or expose them.
"""

from __future__ import annotations

import json

from virtual_context.core.tool_loop import (
    VC_TOOL_NAMES,
    vc_tool_definitions,
    vc_tool_definitions_for_runtime,
)
from virtual_context.types import (
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
)

_FORBIDDEN = ("backfill", "--platform", "backfill_actors", "backfill-actors")


def _assert_clean(surface_name: str, text: str) -> None:
    lowered = text.lower()
    for marker in _FORBIDDEN:
        assert marker not in lowered, (
            f"admin authority marker {marker!r} exposed via {surface_name}"
        )


def test_vc_tool_names_carry_no_admin_authority():
    _assert_clean("VC_TOOL_NAMES", json.dumps(sorted(VC_TOOL_NAMES)))


def test_static_and_runtime_tool_catalogues_carry_no_admin_authority():
    _assert_clean("vc_tool_definitions", json.dumps(vc_tool_definitions()))
    snapshot = SpeakerRosterSnapshot(
        snapshot_id="snap-1",
        entries=(
            SpeakerRosterEntry(
                handle="alex", name="Alex", actor_id="actor:discord:alex",
            ),
        ),
        tenant_id="t1",
        audience_conversation_id="conv-1",
        lifecycle_epoch=1,
    )
    _assert_clean(
        "vc_tool_definitions_for_runtime",
        json.dumps(vc_tool_definitions_for_runtime(roster_snapshot=snapshot)),
    )


def test_mcp_tool_registry_carries_no_admin_authority():
    import asyncio

    from virtual_context.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    rendered = json.dumps(
        [
            {
                "name": t.name,
                "description": t.description or "",
                "schema": t.inputSchema,
            }
            for t in tools
        ],
    )
    assert rendered != "[]"
    _assert_clean("mcp tool registry", rendered)
