"""Automated TUI tests using Textual's headless testing."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]

from conftest import FakeChatProvider
from virtual_context.tui.app import VChatApp
from virtual_context.tui.widgets.budget_bar import BudgetBar
from virtual_context.tui.widgets.chat_view import ChatView
from virtual_context.tui.widgets.input_box import InputBox
from virtual_context.tui.widgets.tag_panel import TagPanel
from virtual_context.tui.widgets.turn_list import TurnList


async def type_text(pilot, text: str) -> None:
    """Type text by pressing individual character keys."""
    for char in text:
        await pilot.press(char)


def make_app() -> VChatApp:
    """Create a VChatApp with mocked provider (no real API calls)."""
    return VChatApp(api_key="fake-key", model="fake-model")


@pytest.mark.asyncio
async def test_widgets_mount():
    """All widgets should mount correctly."""
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        assert app.query_one("#chat-view", ChatView) is not None
        assert app.query_one("#input-box", InputBox) is not None
        assert app.query_one("#tag-panel", TagPanel) is not None
        assert app.query_one("#budget-bar", BudgetBar) is not None
        assert app.query_one("#turn-list", TurnList) is not None
        assert app.query_one("#compaction-log") is not None


@pytest.mark.asyncio
async def test_footer_shows_bindings():
    """Footer should be present with keybinding hints."""
    from textual.widgets import Footer

    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        footer = app.query_one(Footer)
        assert footer is not None


@pytest.mark.asyncio
async def test_enter_submits_message():
    """Typing text and pressing Enter should submit and trigger a turn."""
    app = make_app()
    fake_provider = FakeChatProvider(["Test response from assistant."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "hello")
        await pilot.press("enter")
        await pilot.pause(1.0)

        assert len(app._conversation_history) == 2
        assert app._conversation_history[0].role == "user"
        assert app._conversation_history[0].content == "hello"
        assert app._conversation_history[1].role == "assistant"
        assert "Test response" in app._conversation_history[1].content


@pytest.mark.asyncio
async def test_multiple_turns():
    """Multiple exchanges should accumulate in conversation history."""
    app = make_app()
    fake_provider = FakeChatProvider([
        "First response.",
        "Second response.",
        "Third response.",
    ])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        for msg in ["one", "two", "three"]:
            await type_text(pilot, msg)
            await pilot.press("enter")
            await pilot.pause(1.0)

        # 3 user + 3 assistant = 6 messages
        assert len(app._conversation_history) == 6
        assert len(app._turns) == 3
        assert app._turns[0].turn_number == 1
        assert app._turns[2].turn_number == 3
        assert fake_provider._call_count == 3


@pytest.mark.asyncio
async def test_turn_list_updates():
    """Turn list widget should show entries after each turn."""
    app = make_app()
    fake_provider = FakeChatProvider(["Response one.", "Response two."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "a")
        await pilot.press("enter")
        await pilot.pause(1.0)

        await type_text(pilot, "b")
        await pilot.press("enter")
        await pilot.pause(1.0)

        turn_list = app.query_one("#turn-list", TurnList)
        assert len(turn_list._turns) == 2


@pytest.mark.asyncio
async def test_inspect_turn_modal():
    """Ctrl+I should open the turn inspector modal."""
    app = make_app()
    fake_provider = FakeChatProvider(["Inspect me."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "x")
        await pilot.press("enter")
        await pilot.pause(1.0)

        # Verify api_payload was captured on the turn
        assert len(app._turns) == 1
        payload = app._turns[0].api_payload
        assert "system" in payload
        assert "messages" in payload
        assert len(payload["messages"]) >= 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"].startswith("x")

        # Press Ctrl+I to inspect
        await pilot.press("ctrl+i")
        await pilot.pause(0.5)

        from virtual_context.tui.modals.turn_inspector import TurnInspector
        assert isinstance(app.screen, TurnInspector)

        # Close modal
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert not isinstance(app.screen, TurnInspector)


@pytest.mark.asyncio
async def test_emoji_stripped():
    """Emojis in assistant response should be stripped from display."""
    app = make_app()
    fake_provider = FakeChatProvider(["Hello! How are you?"])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "hi")
        await pilot.press("enter")
        await pilot.pause(1.0)

        chat_view = app.query_one("#chat-view", ChatView)
        assistant_lines = [l for l in chat_view._message_log if "Assistant" in l]
        assert len(assistant_lines) > 0

    # Test the strip function directly
    assert ChatView._strip_emoji("Hello 😊 World 🎉!") == "Hello  World !"


@pytest.mark.asyncio
async def test_turn_navigation():
    """Ctrl+B/F navigate turns in side panel, p/n navigate inside inspector."""
    app = make_app()
    fake_provider = FakeChatProvider(["First.", "Second.", "Third."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        for msg in ["a", "b", "c"]:
            await type_text(pilot, msg)
            await pilot.press("enter")
            await pilot.pause(1.0)

        turn_list = app.query_one("#turn-list", TurnList)
        assert len(turn_list._turns) == 3

        # Latest turn selected by default
        assert turn_list._selected == 2
        assert turn_list.selected_turn.turn_number == 3

        # Ctrl+B moves to previous turn
        await pilot.press("ctrl+b")
        assert turn_list._selected == 1
        assert turn_list.selected_turn.turn_number == 2

        await pilot.press("ctrl+b")
        assert turn_list._selected == 0

        # Can't go below 0
        await pilot.press("ctrl+b")
        assert turn_list._selected == 0

        # Ctrl+F moves forward
        await pilot.press("ctrl+f")
        assert turn_list._selected == 1

        # Inspect selected turn (turn 2)
        await pilot.press("ctrl+i")
        await pilot.pause(0.5)

        from virtual_context.tui.modals.turn_inspector import TurnInspector
        assert isinstance(app.screen, TurnInspector)
        assert app.screen._turn.turn_number == 2

        # p/n navigate inside inspector
        await pilot.press("n")
        await pilot.pause(0.3)
        assert app.screen._turn.turn_number == 3

        await pilot.press("p")
        await pilot.pause(0.3)
        assert app.screen._turn.turn_number == 2

        await pilot.press("escape")
        await pilot.pause(0.3)
        assert not isinstance(app.screen, TurnInspector)


@pytest.mark.asyncio
async def test_token_count_nonzero():
    """Token counts in turns should reflect actual payload size, not 0."""
    app = make_app()
    fake_provider = FakeChatProvider(["This is a test response with some words."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "Tell me about databases")
        await pilot.press("enter")
        await pilot.pause(1.0)

        assert len(app._turns) == 1
        # Token count should be > 0 even without engine (estimated from payload)
        assert app._turns[0].input_tokens > 0


@pytest.mark.asyncio
async def test_turns_in_payload_count():
    """Each turn should track how many turns were bundled in its payload."""
    app = make_app()
    fake_provider = FakeChatProvider(["R1.", "R2.", "R3."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        for msg in ["a", "b", "c"]:
            await type_text(pilot, msg)
            await pilot.press("enter")
            await pilot.pause(1.0)

        # Without engine, all history is sent.
        # Turn 1: 1 user msg → 1 turn in payload
        assert app._turns[0].turns_in_payload == 1
        # Turn 2: 1 user + 1 assistant + 1 user = 3 msgs → 2 turns
        assert app._turns[1].turns_in_payload == 2
        # Turn 3: 5 msgs → 3 turns
        assert app._turns[2].turns_in_payload == 3


@pytest.mark.asyncio
async def test_save_session(tmp_path):
    """Ctrl+S should save session log to a JSON file."""
    import json

    app = make_app()
    fake_provider = FakeChatProvider(["Saved response."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "test")
        await pilot.press("enter")
        await pilot.pause(1.0)

        # Manually test the save function
        from virtual_context.tui.state import save_session, save_turn

        path = save_session(app._turns, directory=str(tmp_path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_turns"] == 1
        assert data["turns"][0]["user_message"] == "test"
        assert data["turns"][0]["turns_in_payload"] == 1

        # Also test single turn save
        turn_path = save_turn(app._turns[0], directory=str(tmp_path))
        assert turn_path.exists()
        turn_data = json.loads(turn_path.read_text())
        assert turn_data["turn_number"] == 1
        assert "api_payload" in turn_data


@pytest.mark.asyncio
async def test_save_turn_from_inspector(tmp_path):
    """Pressing 's' inside the inspector should save the current turn."""
    app = make_app()
    fake_provider = FakeChatProvider(["Response."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "hello")
        await pilot.press("enter")
        await pilot.pause(1.0)

        await pilot.press("ctrl+i")
        await pilot.pause(0.5)

        from virtual_context.tui.modals.turn_inspector import TurnInspector
        assert isinstance(app.screen, TurnInspector)

        # Press 's' to save
        await pilot.press("s")
        await pilot.pause(0.3)

        # Verify the file was created
        import json
        from pathlib import Path
        save_path = Path("vc-turn-1.json")
        assert save_path.exists(), "vc-turn-1.json should have been created"
        data = json.loads(save_path.read_text())
        assert data["turn_number"] == 1
        assert data["user_message"] == "hello"
        save_path.unlink()  # cleanup

        await pilot.press("escape")


@pytest.mark.asyncio
async def test_no_submit_while_streaming():
    """Pressing Enter during streaming should be ignored."""
    app = make_app()

    class SlowProvider:
        api_key = "fake"
        model = "fake"
        _call_count = 0

        def stream_message(self, system, messages, max_tokens=4096):
            self._call_count += 1
            import time
            for word in ["Slow", " ", "response"]:
                time.sleep(0.15)
                yield word

    fake_provider = SlowProvider()

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "a")
        await pilot.press("enter")

        # Immediately try to send another (should be ignored — still streaming)
        await pilot.pause(0.05)
        await type_text(pilot, "b")
        await pilot.press("enter")

        await pilot.pause(1.5)

        # Only 1 provider call should have been made
        assert fake_provider._call_count == 1


@pytest.mark.asyncio
async def test_replay_mode():
    """Replay prompts should be sent sequentially on startup."""
    prompts = ["first question", "second question", "third question"]
    app = VChatApp(api_key="fake-key", model="fake-model", replay_prompts=prompts)
    fake_provider = FakeChatProvider(["Reply 1.", "Reply 2.", "Reply 3."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider
        app.engine = None

        # Wait for all replay turns to complete
        await pilot.pause(4.0)

        assert len(app._conversation_history) == 6  # 3 user + 3 assistant
        assert app._conversation_history[0].content == "first question"
        assert app._conversation_history[2].content == "second question"
        assert app._conversation_history[4].content == "third question"
        assert fake_provider._call_count == 3


@pytest.mark.asyncio
async def test_load_replay_from_session_json(tmp_path):
    """load_replay_prompts should extract user_message from vc-session.json."""
    import json
    from virtual_context.tui.state import load_replay_prompts

    session = {
        "total_turns": 2,
        "turns": [
            {"user_message": "hello", "assistant_message": "hi"},
            {"user_message": "bye", "assistant_message": "later"},
        ],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session))

    prompts = load_replay_prompts(path)
    assert prompts == ["hello", "bye"]


@pytest.mark.asyncio
async def test_load_replay_from_text_file(tmp_path):
    """load_replay_prompts should read one prompt per line from plain text."""
    from virtual_context.tui.state import load_replay_prompts

    path = tmp_path / "prompts.txt"
    path.write_text("tell me about dogs\n\nwhat about cats\nhow about fish\n")

    prompts = load_replay_prompts(path)
    assert prompts == ["tell me about dogs", "what about cats", "how about fish"]


def test_turn_list_selection_tracking():
    """Selection should track correctly across many turns."""
    from virtual_context.tui.widgets.turn_list import TurnList
    from virtual_context.tui.state import TurnRecord
    from virtual_context.types import AssembledContext

    tl = TurnList()

    for i in range(12):
        tl.add_turn(TurnRecord(
            turn_number=i + 1,
            user_message=f"msg {i}",
            assistant_message=f"reply {i}",
            assembled=AssembledContext(),
            turns_in_payload=1,
        ))

    # After adding 12 turns, selected should be the last one
    assert tl._selected == 11
    assert tl.selected_turn.turn_number == 12

    # Navigate all the way to the top
    for _ in range(11):
        tl.select_prev()
    assert tl._selected == 0
    assert tl.selected_turn.turn_number == 1

    # Can't go below 0
    tl.select_prev()
    assert tl._selected == 0

    # Navigate back to the end
    for _ in range(11):
        tl.select_next()
    assert tl._selected == 11
    assert tl.selected_turn.turn_number == 12

    # Can't go past end
    tl.select_next()
    assert tl._selected == 11


@pytest.mark.regression("BUG-002")
@pytest.mark.asyncio
async def test_tag_panel_updates_after_turn_complete():
    """BUG-002: Tag panel must update after on_turn_complete, not just inbound.

    With fast providers (Haiku), the tag panel was showing inbound-only tags
    because Static.update() didn't reliably trigger a repaint when called
    from call_from_thread callbacks in quick succession.  The fix uses
    render() override so the compositor always reads current data.
    """
    app = make_app()
    fake_provider = FakeChatProvider(["Test response."])

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = fake_provider

        # Mock engine that returns distinct inbound vs turn-complete tags
        class MockEngine:
            def __init__(self):
                self._turn_tag_index = MockIndex()
                self.config = type("C", (), {"context_window": 10000})()

            def on_message_inbound(self, msg, history):
                from virtual_context.types import AssembledContext
                ctx = AssembledContext()
                ctx.matched_tags = ["inbound-tag"]
                ctx.temporal = False
                return ctx

            def filter_history(self, history, current_tags=None, temporal=False):
                return history

            def on_turn_complete(self, history):
                from virtual_context.types import TurnTagEntry
                # Simulate engine tagging the user+assistant pair with richer tags
                self._turn_tag_index.add(TurnTagEntry(
                    turn_number=0,
                    message_hash="abc",
                    tags=["complete-tag-a", "complete-tag-b"],
                    primary_tag="complete-tag-a",
                ))
                return None

        class MockIndex:
            def __init__(self):
                self._entries = []

            def add(self, entry):
                self._entries.append(entry)

            @property
            def entries(self):
                return self._entries

        app.engine = MockEngine()

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "hello")
        await pilot.press("enter")
        await pilot.pause(1.5)

        # Turn should have the turn-complete tags (merged with inbound)
        assert len(app._turns) == 1
        turn_tags = app._turns[0].tags
        assert "complete-tag-a" in turn_tags
        assert "complete-tag-b" in turn_tags

        # Tag panel should reflect the turn-complete tags, not just inbound
        tag_panel = app.query_one("#tag-panel", TagPanel)
        panel_tag_names = [name for name, _score in tag_panel._tags]
        assert "complete-tag-a" in panel_tag_names, (
            f"Tag panel should show turn-complete tags, got: {panel_tag_names}"
        )


@pytest.mark.asyncio
async def test_tag_panel_render_method():
    """TagPanel.render() produces correct markup from stored data."""
    tag_panel = TagPanel()
    # Before any tags
    assert "No tags yet" in tag_panel.render()

    # After setting tags
    tag_panel.update_tags([("database", 0.8), ("api", 0.3)])
    rendered = tag_panel.render()
    assert "database" in rendered
    assert "api" in rendered
    assert "green" in rendered  # 0.8 >= 0.7
    assert "red" in rendered    # 0.3 < 0.4


@pytest.mark.asyncio
async def test_streaming_flag_reset_on_error():
    """_streaming must reset even if _execute_turn throws (try/finally guard)."""
    app = make_app()

    class FailingProvider:
        api_key = "fake"
        model = "fake"
        _call_count = 0

        def stream_message(self, system, messages, max_tokens=4096):
            self._call_count += 1
            raise RuntimeError("Simulated API failure")

    async with app.run_test(size=(120, 40)) as pilot:
        app.provider = FailingProvider()
        app.engine = None

        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

        await type_text(pilot, "hello")
        await pilot.press("enter")
        await pilot.pause(1.0)

        # _streaming must be False so user can send another message
        assert app._streaming is False
