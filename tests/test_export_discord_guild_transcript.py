from __future__ import annotations

import pytest

from scripts.export_discord_guild_transcript_20260802 import (
    ExportError,
    _channels,
)


class _GuildChannelClient:
    def __init__(self, *, unknown_type: bool = False) -> None:
        self.calls: list[str] = []
        self.unknown_type = unknown_type

    def get(self, path: str, **_params):
        self.calls.append(path)
        if path == "/guilds/guild-1/channels":
            rows = [
                {"id": "100", "type": 0, "name": "text"},
                {"id": "200", "type": 2, "name": "voice"},
                {"id": "300", "type": 5, "name": "announcements"},
                {"id": "400", "type": 13, "name": "stage"},
                {"id": "500", "type": 15, "name": "forum"},
                {"id": "600", "type": 16, "name": "media"},
                {"id": "700", "type": 4, "name": "category"},
            ]
            if self.unknown_type:
                rows.append({"id": "999", "type": 99, "name": "future"})
            return rows
        if path == "/guilds/guild-1/threads/active":
            return {"threads": [{"id": "801", "type": 11, "name": "active"}]}
        archived = {
            "/channels/100/threads/archived/public": [
                {"id": "810", "type": 11, "name": "text-public"},
            ],
            "/channels/100/threads/archived/private": [
                {"id": "811", "type": 12, "name": "text-private"},
            ],
            "/channels/300/threads/archived/public": [
                {"id": "812", "type": 10, "name": "announcement-thread"},
            ],
            "/channels/500/threads/archived/public": [
                {"id": "813", "type": 11, "name": "forum-post"},
            ],
            "/channels/600/threads/archived/public": [
                {"id": "814", "type": 11, "name": "media-post"},
            ],
        }
        if path in archived:
            return {"threads": archived[path], "has_more": False}
        raise AssertionError(path)


def test_export_includes_all_known_message_bearing_guild_surfaces() -> None:
    client = _GuildChannelClient()
    channels = _channels(client, "guild-1")
    assert {channel["id"] for channel in channels} == {
        "100", "200", "300", "400", "801", "810", "811", "812", "813", "814",
    }
    assert all(channel["guild_id"] == "guild-1" for channel in channels)
    assert "/channels/500/threads/archived/private" not in client.calls
    assert "/channels/600/threads/archived/private" not in client.calls


def test_export_refuses_unknown_guild_channel_type() -> None:
    with pytest.raises(ExportError, match="unknown guild channel type"):
        _channels(_GuildChannelClient(unknown_type=True), "guild-1")
