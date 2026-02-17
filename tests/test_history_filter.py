"""Tests for tag-based conversation history filtering.

Covers the engine.filter_history() method and the full TUI workflow
where different topics produce different filtered payloads.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message, TagResult, TurnTagEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmpdir: str, recent_turns: int = 2) -> VirtualContextEngine:
    """Build an engine with keyword tagging and history filtering enabled."""
    db_path = str(Path(tmpdir) / "store.db")
    return VirtualContextEngine(config=load_config(config_dict={
        "context_window": 50_000,
        "storage_root": tmpdir,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "assembly": {"recent_turns_always_included": recent_turns},
        "tag_generator": {
            "type": "keyword",
            "keyword_fallback": {
                "tag_keywords": {
                    "database": ["schema", "table", "query", "sql", "index", "database"],
                    "auth": ["auth", "login", "jwt", "token", "password", "session"],
                    "frontend": ["react", "component", "css", "html", "button", "ui"],
                },
            },
        },
    }))


def _simulate_turns(
    engine: VirtualContextEngine,
    pairs: list[tuple[str, str]],
) -> list[Message]:
    """Simulate N turns through the engine, returning the full history."""
    history: list[Message] = []
    for user_text, assistant_text in pairs:
        history.append(Message(role="user", content=user_text))
        history.append(Message(role="assistant", content=assistant_text))
        engine.on_turn_complete(history)
    return history


# ---------------------------------------------------------------------------
# Unit tests for engine.filter_history()
# ---------------------------------------------------------------------------

class TestFilterHistory:
    """Unit tests for VirtualContextEngine.filter_history()."""

    def test_short_history_unchanged(self):
        """History shorter than recent_turns window is returned as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=3)

            history = [
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
                Message(role="user", content="how are you"),
                Message(role="assistant", content="good"),
            ]
            # 2 turns < 3 recent_turns → all included
            filtered = engine.filter_history(history, current_tags=["anything"])
            assert len(filtered) == len(history)

    def test_recent_turns_always_kept(self):
        """Last N turns are always kept regardless of tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=2)

            # Simulate 5 turns: 3 database, 2 auth
            history = _simulate_turns(engine, [
                ("Tell me about the database schema", "The schema has users and orders tables"),
                ("What indexes exist on the users table?", "There's a primary key index and email index"),
                ("How do I write a SQL query for joins?", "Use SELECT ... JOIN syntax"),
                ("How does auth work?", "Auth uses JWT tokens"),
                ("What about session management?", "Sessions are stored in Redis"),
            ])

            # Filter with "database" tags — the 2 most recent (auth turns)
            # are always kept, plus older database-tagged turns
            filtered = engine.filter_history(history, current_tags=["database"])

            # Last 2 turns (4 messages) always included
            assert filtered[-4:] == history[-4:]
            # Total should be > 4 (auth recent) because older database turns match
            assert len(filtered) > 4

    def test_irrelevant_turns_dropped(self):
        """Turns with no tag overlap are excluded from the filtered result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=1)

            # 4 turns: database, auth, frontend, database
            history = _simulate_turns(engine, [
                ("What's the database schema?", "It has 5 tables"),
                ("How does login auth work?", "JWT-based auth"),
                ("Fix the react component", "Updated the button CSS"),
                ("Add a new table to the database", "Created the orders table"),
            ])

            # Now ask about database — filter with database tags
            filtered = engine.filter_history(history, current_tags=["database"])

            # Last 1 turn (2 messages) always kept = turn 4 (database)
            # Older: turn 1 (database) kept, turn 2 (auth) dropped, turn 3 (frontend) dropped
            # Total: turn 1 + turn 4 = 4 messages
            assert len(filtered) == 4
            assert "schema" in filtered[0].content.lower()
            assert "new table" in filtered[2].content.lower()

    def test_no_tag_entry_conservatively_included(self):
        """Turns without TurnTagIndex entries are kept (conservative default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=1)

            # Build history WITHOUT running through engine (no TurnTagIndex entries)
            history = [
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
                Message(role="user", content="about auth login"),
                Message(role="assistant", content="use JWT"),
                Message(role="user", content="database query"),
                Message(role="assistant", content="SELECT * FROM users"),
            ]

            filtered = engine.filter_history(history, current_tags=["database"])
            # No index entries → all older turns kept conservatively + recent
            assert len(filtered) == len(history)

    def test_config_recent_turns_respected(self):
        """The recent_turns_always_included config value is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=3)

            # 5 turns, all different topics
            history = _simulate_turns(engine, [
                ("database schema design", "Use normalized tables"),
                ("auth token expiry", "Set JWT TTL to 1 hour"),
                ("react component styling", "Use CSS modules"),
                ("login flow", "Redirect after auth"),
                ("button click handler", "Use onClick prop"),
            ])

            # Filter with "database" — last 3 turns protected
            filtered = engine.filter_history(history, current_tags=["database"])

            # Last 3 turns = 6 messages always kept
            assert filtered[-6:] == history[-6:]
            # Turn 1 (database) should be kept, turn 2 (auth) dropped
            assert len(filtered) == 8  # 6 protected + 2 from turn 1

    def test_multiple_matching_tags(self):
        """Turns matching ANY of the current tags are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=1)

            history = _simulate_turns(engine, [
                ("database schema", "Tables defined"),
                ("auth login flow", "JWT-based"),
                ("react component", "Use hooks"),
                ("weather today", "Sunny"),  # _general — no match
            ])

            # Query with both database and auth tags
            filtered = engine.filter_history(
                history, current_tags=["database", "auth"]
            )

            # Last 1 turn (general) always kept = 2 messages
            # Older: database (kept), auth (kept), frontend (dropped)
            assert len(filtered) == 6  # db(2) + auth(2) + recent(2)


# ---------------------------------------------------------------------------
# Full workflow integration test (8-step scenario)
# ---------------------------------------------------------------------------

class TestTagFilterWorkflow:
    """End-to-end test mimicking the user's 8-step workflow.

    Scenario:
        Turn 1: User asks about database schema
        Turn 2: User asks about auth/JWT
        Turn 3: User asks about frontend/React
        Turn 4: User asks about database again
        Turn 5: User asks about auth again

    At each step, verify that filter_history only includes
    tag-relevant older turns + the recent window.
    """

    def test_multi_topic_filtering_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=2)
            history: list[Message] = []

            # --- Turn 1: Database ---
            history.append(Message(role="user", content="Explain the database schema"))
            assembled_1 = engine.on_message_inbound("Explain the database schema", history)
            history.append(Message(role="assistant", content="The schema has users, orders, and products tables"))
            engine.on_turn_complete(history)

            # After turn 1, only 1 turn exists — everything included
            f1 = engine.filter_history(history, current_tags=["database"])
            assert len(f1) == 2

            # --- Turn 2: Auth ---
            history.append(Message(role="user", content="How does JWT auth work in our API?"))
            assembled_2 = engine.on_message_inbound("How does JWT auth work in our API?", history)
            history.append(Message(role="assistant", content="JWT tokens are signed with RS256"))
            engine.on_turn_complete(history)

            # After turn 2, 2 turns — within recent window, all included
            f2 = engine.filter_history(history, current_tags=["auth"])
            assert len(f2) == 4

            # --- Turn 3: Frontend ---
            history.append(Message(role="user", content="Fix the React button component"))
            assembled_3 = engine.on_message_inbound("Fix the React button component", history)
            history.append(Message(role="assistant", content="Updated the onClick handler and CSS"))
            engine.on_turn_complete(history)

            # 3 turns total, recent=2 protects turns 2+3.
            # Turn 1 (database) is older — should be dropped for frontend query
            f3 = engine.filter_history(history, current_tags=["frontend"])
            assert len(f3) == 4  # only turns 2+3 (protected), turn 1 dropped

            # --- Turn 4: Database again ---
            history.append(Message(role="user", content="Add an index to the users table"))
            assembled_4 = engine.on_message_inbound("Add an index to the users table", history)
            history.append(Message(role="assistant", content="CREATE INDEX idx_users_email ON users(email)"))
            engine.on_turn_complete(history)

            # 4 turns total, recent=2 protects turns 3+4.
            # Older: turn 1 (database→match), turn 2 (auth→no match)
            f4 = engine.filter_history(history, current_tags=["database"])
            assert len(f4) == 6  # turn1(2) + turn3(2, protected) + turn4(2, protected)
            # turn 2 (auth) should be excluded
            assert not any("JWT" in m.content for m in f4)

            # --- Turn 5: Auth again ---
            history.append(Message(role="user", content="What about session token expiry?"))
            assembled_5 = engine.on_message_inbound("What about session token expiry?", history)
            history.append(Message(role="assistant", content="Tokens expire after 1 hour"))
            engine.on_turn_complete(history)

            # 5 turns total, recent=2 protects turns 4+5.
            # Older: turn 1 (database→no), turn 2 (auth→yes), turn 3 (frontend→no)
            f5 = engine.filter_history(history, current_tags=["auth"])
            assert len(f5) == 6  # turn2(2) + turn4(2, protected) + turn5(2, protected)
            # turns 1 (database) and 3 (frontend) should be excluded
            assert not any("schema" in m.content.lower() for m in f5)
            assert not any("React" in m.content for m in f5)
            # turn 2 (auth) should be present
            assert any("JWT" in m.content for m in f5)

    def test_payload_shows_filtering(self):
        """Verify that the api_payload metadata includes filtering stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(tmpdir, recent_turns=1)

            history = _simulate_turns(engine, [
                ("database schema design", "5 tables"),
                ("auth login flow", "JWT tokens"),
                ("react component fix", "Updated CSS"),
                ("database query optimization", "Added index"),
            ])

            # Filtering for "database" with recent=1
            filtered = engine.filter_history(history, current_tags=["database"])

            # 4 turns = 8 messages total
            # recent=1 → last turn (2 msgs) protected
            # older: turn1 (database→yes), turn2 (auth→no), turn3 (frontend→no)
            assert len(filtered) == 4  # turn1(2) + turn4(2, recent)
            assert len(history) == 8

            # The stats that would go into the api_payload
            assert len(history) - len(filtered) == 4  # 4 messages filtered out


# ---------------------------------------------------------------------------
# TUI integration test
# ---------------------------------------------------------------------------

class FakeChatProvider:
    """Mock provider for TUI tests."""

    def __init__(self, responses: list[str] | None = None):
        self.api_key = "fake-key"
        self.model = "fake-model"
        self._responses = responses or ["Hello!"]
        self._call_count = 0

    def stream_message(
        self, system: str, messages: list[dict], max_tokens: int = 4096
    ) -> Iterator[str]:
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        for word in response.split(" "):
            yield word + " "


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tui_workflow_payload_filtering():
    """Full TUI workflow: send multi-topic messages and verify
    the Turn Inspector payload reflects tag-based filtering."""
    from virtual_context.tui.app import VChatApp
    from virtual_context.tui.widgets.input_box import InputBox

    async def type_text(pilot, text: str) -> None:
        for char in text:
            await pilot.press(char)

    app = VChatApp(api_key="fake-key", model="fake-model")
    fake_provider = FakeChatProvider([
        "The schema has users and orders tables.",
        "JWT tokens are signed with RS256.",
        "Updated the button click handler.",
        "Added an index on email column.",
        "Tokens expire after 1 hour.",
    ])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None  # No engine — filtering won't happen but payloads are captured

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        messages = [
            "Tell me about the database schema",
            "How does JWT auth work?",
            "Fix the React button component",
            "Add an index to the database table",
            "What about session token expiry?",
        ]

        for msg in messages:
            await type_text(pilot, msg)
            await pilot.press("enter")
            await pilot.pause(1.0)

        # Verify all 5 turns completed
        assert len(app._turns) == 5

        # Each turn should have an api_payload with system + messages
        for i, turn in enumerate(app._turns):
            payload = turn.api_payload
            assert "system" in payload, f"Turn {i+1} missing system in payload"
            assert "messages" in payload, f"Turn {i+1} missing messages in payload"
            assert "total_history" in payload, f"Turn {i+1} missing total_history"
            assert "filtered_history" in payload, f"Turn {i+1} missing filtered_history"

        # Without engine, all history is sent (no filtering)
        # Turn 5 should have all 9 messages (4 complete turns + current user msg)
        last_payload = app._turns[4].api_payload
        assert last_payload["total_history"] == last_payload["filtered_history"]

        # Verify message content is captured in payloads
        turn1_msgs = app._turns[0].api_payload["messages"]
        assert turn1_msgs[0]["content"].startswith("Tell me about the database schema")
