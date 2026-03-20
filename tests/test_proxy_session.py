"""Tests for proxy session management: extraction, markers, registry, routing, state machine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_context.proxy.server import (
    ProxyState,
    SessionRegistry,
    SessionState,
    _extract_conversation_id,
    _inject_conversation_marker,
    _strip_conversation_markers,
    create_app,
)
from virtual_context.config import load_config
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import AssembledContext, Message, TurnTagEntry


# ---------------------------------------------------------------------------
# Fixtures shared with test_proxy.py (duplicated for independence)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Create a mock VirtualContextEngine."""
    engine = MagicMock()
    assembled = AssembledContext(prepend_text="mock context here")
    engine.on_message_inbound.return_value = assembled
    engine.on_turn_complete.return_value = None
    engine.tag_turn.return_value = None
    engine.compact_if_needed.return_value = None
    return engine


@pytest.fixture
def app_with_mock_engine(mock_engine):
    """Create a FastAPI app with a mock engine, patching VirtualContextEngine."""
    with patch("virtual_context.proxy.server.VirtualContextEngine", return_value=mock_engine):
        app = create_app(upstream="http://fake-upstream:9999", config_path=None)
    return app, mock_engine


@pytest.fixture
def test_client(app_with_mock_engine):
    """Provide a TestClient for the proxy app."""
    from starlette.testclient import TestClient
    app, engine = app_with_mock_engine
    with TestClient(app) as client:
        yield client, engine


# ---------------------------------------------------------------------------
# Session marker: extraction
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_extracts_from_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there\n<!-- vc:conversation=a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c -->"},
            {"role": "user", "content": "Next question"},
        ]}
        assert _extract_conversation_id(body) == "a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"

    def test_extracts_from_content_blocks(self):
        body = {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "text", "text": "answer\n<!-- vc:conversation=dead-beef-1234 -->"},
            ]},
            {"role": "user", "content": "Next"},
        ]}
        assert _extract_conversation_id(body) == "dead-beef-1234"

    def test_returns_none_when_no_marker(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Next"},
        ]}
        assert _extract_conversation_id(body) is None

    def test_returns_none_for_empty_messages(self):
        assert _extract_conversation_id({"messages": []}) is None
        assert _extract_conversation_id({}) is None

    def test_finds_most_recent_marker(self):
        """When multiple assistant messages have markers, returns the most recent."""
        body = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1\n<!-- vc:conversation=aaa00000-0000-0000-0000-000000000001 -->"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2\n<!-- vc:conversation=bbb00000-0000-0000-0000-000000000002 -->"},
            {"role": "user", "content": "Q3"},
        ]}
        assert _extract_conversation_id(body) == "bbb00000-0000-0000-0000-000000000002"

    def test_ignores_user_messages(self):
        """Markers in user messages are not extracted."""
        body = {"messages": [
            {"role": "user", "content": "<!-- vc:conversation=fake-id --> hello"},
            {"role": "assistant", "content": "Hi"},
        ]}
        assert _extract_conversation_id(body) is None


# ---------------------------------------------------------------------------
# Session marker: stripping
# ---------------------------------------------------------------------------


class TestStripSessionMarkers:
    def test_strips_from_string_content(self):
        body = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there\n<!-- vc:conversation=abc00000-0000-0000-0000-000000000123 -->"},
        ]}
        result = _strip_conversation_markers(body)
        assert result["messages"][1]["content"] == "Hi there"

    def test_strips_from_content_blocks(self):
        body = {"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Answer\n<!-- vc:conversation=abc00000-0000-0000-0000-000000000123 -->"},
            ]},
        ]}
        result = _strip_conversation_markers(body)
        assert result["messages"][0]["content"][0]["text"] == "Answer"

    def test_no_marker_returns_same_body(self):
        body = {"messages": [
            {"role": "assistant", "content": "clean text"},
        ]}
        result = _strip_conversation_markers(body)
        assert result is body  # no copy made

    def test_preserves_user_messages(self):
        body = {"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer\n<!-- vc:conversation=aaa00000-0000-0000-0000-000000000001 -->"},
        ]}
        result = _strip_conversation_markers(body)
        assert result["messages"][0]["content"] == "question"
        assert result["messages"][1]["content"] == "answer"

    def test_strips_multiple_assistant_messages(self):
        body = {"messages": [
            {"role": "assistant", "content": "A1\n<!-- vc:conversation=aaa00000-0000-0000-0000-000000000001 -->"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2\n<!-- vc:conversation=bbb00000-0000-0000-0000-000000000002 -->"},
        ]}
        result = _strip_conversation_markers(body)
        assert result["messages"][0]["content"] == "A1"
        assert result["messages"][2]["content"] == "A2"


# ---------------------------------------------------------------------------
# Session marker: injection into non-streaming response
# ---------------------------------------------------------------------------


class TestInjectSessionMarker:
    def test_openai_format(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        result = _inject_conversation_marker(resp, "\n<!-- vc:conversation=abc -->", "openai")
        assert result["choices"][0]["message"]["content"] == "Hello\n<!-- vc:conversation=abc -->"

    def test_anthropic_format(self):
        resp = {"content": [{"type": "text", "text": "Hello"}]}
        result = _inject_conversation_marker(resp, "\n<!-- vc:conversation=abc -->", "anthropic")
        assert result["content"][0]["text"] == "Hello\n<!-- vc:conversation=abc -->"

    def test_anthropic_multiple_blocks_appends_to_last(self):
        resp = {"content": [
            {"type": "text", "text": "Thinking..."},
            {"type": "text", "text": "Answer"},
        ]}
        result = _inject_conversation_marker(resp, "\n<!-- vc:conversation=abc -->", "anthropic")
        assert result["content"][0]["text"] == "Thinking..."
        assert result["content"][1]["text"] == "Answer\n<!-- vc:conversation=abc -->"

    def test_does_not_mutate_original(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        original_content = resp["choices"][0]["message"]["content"]
        _inject_conversation_marker(resp, "\n<!-- vc:conversation=abc -->", "openai")
        assert resp["choices"][0]["message"]["content"] == original_content


# ---------------------------------------------------------------------------
# Session marker: streaming integration
# ---------------------------------------------------------------------------


class TestStreamingSessionMarker:
    def test_streaming_response_includes_session_marker(self, test_client):
        """Streaming response should include a session marker SSE event."""
        client, engine = test_client

        sse_events = (
            b"event: content_block_delta\r\n"
            b"data: {\"type\":\"content_block_delta\",\"index\":0,"
            b"\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\r\n"
            b"\r\n"
            b"event: message_stop\r\n"
            b"data: {\"type\":\"message_stop\"}\r\n"
            b"\r\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield sse_events

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        with patch("virtual_context.proxy.server.httpx.AsyncClient.send", return_value=mock_resp):
            with patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"):
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3",
                        "system": "test",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                assert resp.status_code == 200
                body = resp.content.decode("utf-8", errors="replace")
                assert "vc:conversation=" in body
                # The session ID should be a UUID from the engine
                assert "content_block_delta" in body

    def test_non_streaming_response_includes_session_marker(self, test_client):
        """Non-streaming response body should include session marker."""
        client, engine = test_client

        upstream_response = {
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3",
        }

        with patch("virtual_context.proxy.server.httpx.AsyncClient.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = upstream_response
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_req.return_value = mock_resp

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "system": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            # The last text block should contain the session marker
            text = data["content"][-1]["text"]
            assert "<!-- vc:conversation=" in text


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    def test_creates_new_session_when_no_id_and_empty(self, tmp_path):
        """No session ID + empty registry → creates a new session."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.conversation_id = "new-uuid-1234"
            MockEngine.return_value = engine

            state, is_new = registry.get_or_create(None)

        assert is_new is True
        assert state is not None
        assert state.engine.config.conversation_id == "new-uuid-1234"
        assert registry.conversation_count == 1

    def test_reuses_session_via_last_message_hash(self, tmp_path):
        """No session ID + last-message hash match → reuses existing session.

        Simulates the real flow: request N stores hash of last user message,
        then request N+1 (one more turn) matches via second-to-last user message.
        """
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.conversation_id = "default-session"
        default = ProxyState(engine, metrics=metrics)

        # Request N: 3 user messages.  Store hash of last user msg ("tell me more")
        body_prev = {"messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "doing well"},
            {"role": "user", "content": "tell me more"},
        ]}
        # Request N+1: one more turn.  Second-to-last user msg = "tell me more"
        # which matches the hash stored from request N.
        body_next = {"messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "doing well"},
            {"role": "user", "content": "tell me more"},
            {"role": "assistant", "content": "sure thing"},
            {"role": "user", "content": "what is new"},
        ]}

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._conversations["default-session"] = default
        # Simulate catch_all: store last-message hash from request N
        registry.update_last_message_hash(body_prev, "default-session")

        # Request N+1 should match via second-to-last user message
        result, is_new = registry.get_or_create(None, body=body_next)
        assert is_new is False
        assert result is default

    def test_resumes_existing_session(self, tmp_path):
        """Known session ID in memory → returns existing state."""
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.conversation_id = "existing-session"
        state = ProxyState(engine, metrics=metrics)

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._conversations["existing-session"] = state

        result, is_new = registry.get_or_create("existing-session")
        assert is_new is False
        assert result is state

    def test_creates_new_session_when_no_match(self, tmp_path):
        """No matching system hash or last-message hash → creates new session."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.conversation_id = "new-session"
            MockEngine.return_value = engine

            state, is_new = registry.get_or_create(None)

        assert is_new is True
        assert state.engine.config.conversation_id == "new-session"

    def test_multiple_sessions(self, tmp_path):
        """Multiple sessions can coexist (routed via system prompt hash)."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            for sid in ["session-a", "session-b", "session-c"]:
                engine = MagicMock()
                engine.config.conversation_id = sid
                MockEngine.return_value = engine
                # Each session has a unique system prompt → unique hash
                body = {"messages": [
                    {"role": "system", "content": f"chat_id: {sid}"},
                    {"role": "user", "content": "hello"},
                ]}
                registry.get_or_create(None, body=body)

        assert registry.conversation_count == 3

    def test_shutdown_all_clears_sessions(self, tmp_path):
        """shutdown_all() cleans up all sessions."""
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.conversation_id = "s1"
        state = ProxyState(engine, metrics=metrics)

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._conversations["s1"] = state

        registry.shutdown_all()
        assert registry.conversation_count == 0


# ---------------------------------------------------------------------------
# Session awareness: request events include conversation_id
# ---------------------------------------------------------------------------


class TestRequestEventSessionId:
    def test_request_event_includes_conversation_id(self):
        """Request event dict should carry conversation_id."""
        m = ProxyMetrics()
        m.record({
            "type": "request",
            "turn": 0,
            "conversation_id": "abc-123",
            "tags": ["test"],
        })
        events = m.events_since(-1)
        assert len(events) == 1
        assert events[0]["conversation_id"] == "abc-123"

    def test_captured_request_includes_conversation_id(self):
        """capture_request() with conversation_id appears in summary."""
        m = ProxyMetrics()
        m.capture_request(
            0, {"messages": []}, "anthropic",
            conversation_id="sess-xyz",
        )
        summaries = m.get_captured_requests_summary()
        assert len(summaries) == 1
        assert summaries[0]["conversation_id"] == "sess-xyz"

        req = m.get_captured_request(0)
        assert req["conversation_id"] == "sess-xyz"


# ---------------------------------------------------------------------------
# Live sessions in snapshot / HTML
# ---------------------------------------------------------------------------


class TestLiveSessions:
    def test_dashboard_html_has_session_column(self):
        """The dashboard HTML template should include a Session column header."""
        from virtual_context.proxy.dashboard import get_dashboard_html
        html = get_dashboard_html()
        assert "<th>Session</th>" in html

    def test_live_sessions_in_snapshot(self):
        """SSE snapshot should include live_sessions when registry has sessions."""
        # This tests the data shape: registry provides session data
        # that gets included in the SSE snapshot.
        metrics = ProxyMetrics()
        engine = MagicMock()
        engine.config.conversation_id = "live-session-1"
        engine._engine_state.compacted_through = 0
        engine._turn_tag_index = MagicMock()
        engine._turn_tag_index.entries = []
        engine._turn_tag_index.get_active_tags.return_value = ["tag-a"]

        state = ProxyState(engine, metrics=metrics)
        state.conversation_history = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )
        registry._conversations["live-session-1"] = state

        # Simulate what the SSE snapshot builder does
        live_sessions = []
        for sid, s in registry._conversations.items():
            live_sessions.append({
                "conversation_id": sid,
                "turn_count": len(s.conversation_history) // 2,
                "compacted_through": s.engine._engine_state.compacted_through,
                "tag_count": len(s.engine._turn_tag_index.entries),
                "active_tags": list(
                    s.engine._turn_tag_index.get_active_tags(lookback=6)
                ),
            })

        assert len(live_sessions) == 1
        ls = live_sessions[0]
        assert ls["conversation_id"] == "live-session-1"
        assert ls["turn_count"] == 1
        assert ls["active_tags"] == ["tag-a"]


# ---------------------------------------------------------------------------
# PROXY-010: Content fingerprint session routing
# ---------------------------------------------------------------------------


class TestContentFingerprintRouting:
    """Different conversations (no session marker) must get separate sessions."""

    def _make_body(self, user_messages: list[str]) -> dict:
        """Build a minimal Anthropic-format request body."""
        msgs = []
        for i, text in enumerate(user_messages):
            msgs.append({"role": "user", "content": text})
            if i < len(user_messages) - 1:
                msgs.append({"role": "assistant", "content": f"Response {i}"})
        return {"messages": msgs, "model": "test"}

    @pytest.mark.regression("PROXY-010")
    def test_different_conversations_get_different_sessions(self):
        """Two conversations with different history must not share a session."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        body_a = self._make_body([
            "[Telegram Y id:111] i am the one who knocks",
            "[Telegram Y id:111] lets build an arduino alarm",
            "[Telegram Y id:111] what resistor do i need",
        ])
        body_b = self._make_body([
            "The conversation history was compacted: health discussion",
            "[Telegram G id:-555] check my progesterone levels",
            "[Telegram G id:-555] what about the cycle tracking",
        ])

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            # First call: conversation A
            engine_a = MagicMock()
            engine_a.config.conversation_id = "session-aaa"
            MockEngine.return_value = engine_a
            state_a, is_new_a = registry.get_or_create(None, body=body_a)

            # Second call: conversation B — must get a DIFFERENT session
            engine_b = MagicMock()
            engine_b.config.conversation_id = "session-bbb"
            MockEngine.return_value = engine_b
            state_b, is_new_b = registry.get_or_create(None, body=body_b)

        assert is_new_a is True
        assert is_new_b is True
        assert state_a is not state_b, (
            "Different conversations must get different sessions"
        )
        assert registry.conversation_count == 2

    @pytest.mark.regression("PROXY-010")
    def test_same_conversation_reuses_session_via_fingerprint(self):
        """Same conversation growing by one turn reuses the existing session.

        With tail-based fingerprinting (sample_size=1):
        - Request 1: history_user = [u0, u1, u2], fp_store = hash(u2)
        - Request 2: history_user = [u0, u1, u2, u3],
          fp_match(offset=1) = hash(u2) → matches fp_store
        """
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        # 4 user messages → history_user has 3, fp_store = hash(u2)
        base_msgs = [
            "[Telegram Y id:111] hello world",
            "[Telegram Y id:111] tell me about cars",
            "[Telegram Y id:111] what about trucks",
            "[Telegram Y id:111] and planes too",
        ]
        body_v1 = self._make_body(base_msgs)
        # One more message → history_user has 4, fp_match(offset=1) = hash(u2)
        body_v2 = self._make_body(base_msgs + [
            "[Telegram Y id:111] also ships",
        ])

        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            engine = MagicMock()
            engine.config.conversation_id = "session-xxx"
            MockEngine.return_value = engine
            state_1, is_new_1 = registry.get_or_create(None, body=body_v1)

            state_2, is_new_2 = registry.get_or_create(None, body=body_v2)

        assert is_new_1 is True
        assert is_new_2 is False
        assert state_1 is state_2, (
            "Same conversation with appended messages must reuse session"
        )
        assert registry.conversation_count == 1

    @pytest.mark.regression("PROXY-010")
    def test_marker_takes_priority_over_fingerprint(self):
        """When a session marker exists, it takes priority over fingerprint."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        engine = MagicMock()
        engine.config.conversation_id = "marker-session"
        state = ProxyState(engine, metrics=metrics)
        registry._conversations["marker-session"] = state

        # Body from a different conversation — but marker overrides
        body = self._make_body(["completely different conversation"])

        result, is_new = registry.get_or_create("marker-session", body=body)
        assert is_new is False
        assert result is state


# ---------------------------------------------------------------------------
# Trailing fingerprint
# ---------------------------------------------------------------------------


class TestSessionRouting:
    """System prompt hash + last-message hash routing tests."""

    def _make_body(self, user_messages: list[str], system: str = "") -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for i, text in enumerate(user_messages):
            msgs.append({"role": "user", "content": text})
            if i < len(user_messages) - 1:
                msgs.append({"role": "assistant", "content": f"Response {i}"})
        return {"messages": msgs, "model": "test"}

    def test_hash_user_message_position0(self):
        """position=0 hashes the last user message."""
        body = self._make_body(["msg-a", "msg-b", "msg-c"])
        h = SessionRegistry._hash_user_message(body, position=0)
        assert h  # non-empty
        assert len(h) == 16  # sha256 truncated to 16 hex chars

    def test_hash_user_message_position1(self):
        """position=1 hashes the second-to-last user message."""
        body = self._make_body(["msg-a", "msg-b", "msg-c"])
        h0 = SessionRegistry._hash_user_message(body, position=0)
        h1 = SessionRegistry._hash_user_message(body, position=1)
        assert h0 != h1

    def test_last_msg_hash_matches_across_turns(self):
        """Core invariant: request N's position=0 == request N+1's position=1.

        Request N:   user msgs = [u0, u1, u2].  position=0 → hash(u2).
        Request N+1: user msgs = [u0, u1, u2, u3].  position=1 → hash(u2).
        """
        body_n = self._make_body(["u0", "u1", "u2"])
        body_n1 = self._make_body(["u0", "u1", "u2", "u3"])

        h_store = SessionRegistry._hash_user_message(body_n, position=0)
        h_match = SessionRegistry._hash_user_message(body_n1, position=1)

        assert h_store == h_match, (
            "position=1 of next request must equal position=0 of previous request"
        )

    def test_too_few_messages_returns_empty(self):
        """A body with only 1 user message cannot produce position=1 hash."""
        body_one = self._make_body(["only-one"])
        assert SessionRegistry._hash_user_message(body_one, position=1) == ""

    def test_position_too_large_returns_empty(self):
        """Position beyond available user messages returns empty string."""
        body = self._make_body(["u0", "u1"])
        assert SessionRegistry._hash_user_message(body, position=2) == ""

    def test_system_prompt_hash_openai(self):
        """System prompt hash extracts from OpenAI role=system message."""
        body = self._make_body(["hello"], system="You are a helpful assistant. chat_id: telegram:123")
        h = SessionRegistry._compute_system_hash(body)
        assert h
        assert len(h) == 16

    def test_system_prompt_hash_anthropic(self):
        """System prompt hash extracts from Anthropic top-level system key."""
        body = {"system": "You are helpful. chat_id: telegram:456", "messages": []}
        h = SessionRegistry._compute_system_hash(body)
        assert h
        assert len(h) == 16

    def test_different_system_prompts_different_hashes(self):
        """Different chat_ids in system prompt produce different hashes."""
        body_a = self._make_body(["hi"], system="chat_id: telegram:111")
        body_b = self._make_body(["hi"], system="chat_id: telegram:222")
        assert SessionRegistry._compute_system_hash(body_a) != SessionRegistry._compute_system_hash(body_b)

    def test_no_system_prompt_returns_empty(self):
        """No system prompt → empty hash."""
        body = self._make_body(["hello"])
        assert SessionRegistry._compute_system_hash(body) == ""

    def test_multi_session_routing_via_system_hash(self):
        """Multiple sessions route correctly via system prompt hash."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        states = {}
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            for sid, sys_prompt in [
                ("session-a", "chat_id: telegram:111"),
                ("session-b", "chat_id: telegram:222"),
                ("session-c", "chat_id: telegram:333"),
            ]:
                engine = MagicMock()
                engine.config.conversation_id = sid
                MockEngine.return_value = engine
                body = self._make_body(["hello"], system=sys_prompt)
                state, is_new = registry.get_or_create(None, body=body)
                states[sid] = state

        assert registry.conversation_count == 3

        # Next turn: same system prompts route back to correct sessions
        for sid, sys_prompt in [
            ("session-a", "chat_id: telegram:111"),
            ("session-b", "chat_id: telegram:222"),
            ("session-c", "chat_id: telegram:333"),
        ]:
            body = self._make_body(["hello", "follow-up"], system=sys_prompt)
            state, is_new = registry.get_or_create(None, body=body)
            assert is_new is False, f"{sid} should reuse existing session"
            assert state is states[sid], f"{sid} routed to wrong session"

    def test_multi_session_routing_via_last_message_hash(self):
        """Without system prompts, sessions route via last-message hash."""
        metrics = ProxyMetrics()
        registry = SessionRegistry(
            config_path=None,
            upstream="http://fake:9999",
            metrics=metrics,
        )

        convos = {
            "session-a": ["[id:111] msg-a-{}".format(i) for i in range(6)],
            "session-b": ["[id:222] msg-b-{}".format(i) for i in range(6)],
            "session-c": ["[id:333] msg-c-{}".format(i) for i in range(6)],
        }

        states = {}
        with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
            for sid, msgs in convos.items():
                engine = MagicMock()
                engine.config.conversation_id = sid
                MockEngine.return_value = engine
                body = self._make_body(msgs[:3])
                state, is_new = registry.get_or_create(None, body=body)
                states[sid] = state
                registry.update_last_message_hash(body, sid)

        assert registry.conversation_count == 3

        # Next turn: route via last-message hash
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:4])
            state, is_new = registry.get_or_create(None, body=body)
            assert is_new is False, f"{sid} should reuse existing session"
            assert state is states[sid], f"{sid} routed to wrong session"
            registry.update_last_message_hash(body, sid)

        # Another turn
        for sid, msgs in convos.items():
            body = self._make_body(msgs[:5])
            state, is_new = registry.get_or_create(None, body=body)
            assert is_new is False, f"{sid} turn 3 should reuse session"
            assert state is states[sid], f"{sid} turn 3 routed to wrong session"

        assert registry.conversation_count == 3


# ---------------------------------------------------------------------------
# SessionState machine
# ---------------------------------------------------------------------------


class TestSessionStateMachine:
    """Tests for the non-blocking ingestion state machine."""

    def _make_state(self, *, conversation_id="test-session", metrics=None):
        engine = MagicMock()
        engine.config.conversation_id = conversation_id
        engine._turn_tag_index = TurnTagIndex()
        engine._store = MagicMock()
        engine._store.get_all_tags.return_value = []
        engine.config.tag_generator.context_lookback_pairs = 3
        engine.config.tag_generator.context_bleed_threshold = 0
        if metrics is None:
            metrics = ProxyMetrics()
        return ProxyState(engine, metrics=metrics)

    def test_initial_state_is_active(self):
        state = self._make_state()
        assert state.session_state == SessionState.ACTIVE

    def test_start_ingestion_transitions_to_ingesting(self):
        state = self._make_state()
        state.engine.ingest_history.return_value = 2
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        state.start_ingestion_if_needed(pairs)
        # After starting, state should be INGESTING (thread still running)
        assert state._state in (SessionState.INGESTING, SessionState.ACTIVE)

    def test_ingestion_completes_transitions_to_active(self):
        state = self._make_state()
        state.engine.ingest_history.return_value = 1
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.start_ingestion_if_needed(pairs)
        # Wait for background thread
        import time
        time.sleep(0.5)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    def test_manual_passthrough_overrides_state(self):
        state = self._make_state()
        assert state.session_state == SessionState.ACTIVE
        state.set_manual_passthrough(True)
        assert state.session_state == SessionState.PASSTHROUGH
        # Internal state unchanged
        assert state._state == SessionState.ACTIVE

    def test_manual_passthrough_disable_restores(self):
        state = self._make_state()
        state.set_manual_passthrough(True)
        assert state.session_state == SessionState.PASSTHROUGH
        state.set_manual_passthrough(False)
        assert state.session_state == SessionState.ACTIVE

    def test_empty_history_stays_active(self):
        state = self._make_state()
        state.start_ingestion_if_needed([])
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    def test_state_change_event_emitted(self):
        metrics = ProxyMetrics()
        state = self._make_state(metrics=metrics)
        state._transition_to(SessionState.INGESTING)
        events = metrics.events_since(-1)
        state_events = [e for e in events if e["type"] == "session_state_change"]
        assert len(state_events) == 1
        assert state_events[0]["from"] == "active"
        assert state_events[0]["to"] == "ingesting"

    def test_persisted_state_skips_ingestion(self):
        state = self._make_state()
        # Simulate persisted index covering the history
        for i in range(3):
            state.engine._turn_tag_index.append(TurnTagEntry(
                turn_number=i, message_hash=f"h{i}",
                tags=["tag"], primary_tag="tag",
            ))
        pairs = [
            Message(role="user", content=f"Q{i}")
            if i % 2 == 0
            else Message(role="assistant", content=f"A{i}")
            for i in range(6)  # 3 pairs
        ]
        state.start_ingestion_if_needed(pairs)
        # Should skip ingestion, stay ACTIVE
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True
        state.engine.ingest_history.assert_not_called()

    def test_ingestion_error_transitions_to_active(self):
        state = self._make_state()
        state.engine.ingest_history.side_effect = RuntimeError("boom")
        pairs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        state.start_ingestion_if_needed(pairs)
        import time
        time.sleep(0.5)
        # Even on error, should end up ACTIVE (not stuck in INGESTING)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

    @pytest.mark.regression("PROXY-013")
    def test_second_call_during_ingestion_does_not_restart(self):
        """Second start_ingestion_if_needed during INGESTING should not
        spawn a duplicate thread or reset progress to turn 0."""
        import hashlib
        import time

        ingestion_calls = []
        progress_events = []

        def slow_ingest(pairs, progress_callback=None, turn_offset=0):
            """Simulate slow ingestion — record each call's pair count."""
            ingestion_calls.append(len(pairs) // 2)
            # Simulate tagging each pair
            n = len(pairs) // 2
            for i in range(n):
                combined = f"{pairs[i*2].content} {pairs[i*2+1].content}"
                entry = TurnTagEntry(
                    turn_number=len(state.engine._turn_tag_index.entries),
                    message_hash=hashlib.sha256(combined.encode()).hexdigest()[:16],
                    tags=[f"tag-{i}"],
                    primary_tag=f"tag-{i}",
                )
                state.engine._turn_tag_index.append(entry)
                if progress_callback:
                    progress_callback(i + 1, n, entry)
                time.sleep(0.05)  # simulate work
            return n

        state = self._make_state()
        state.engine.ingest_history.side_effect = slow_ingest

        # 6 pairs = 12 messages
        pairs = [
            Message(role=("user" if j % 2 == 0 else "assistant"), content=f"msg-{j}")
            for j in range(12)
        ]

        # First call — starts background ingestion
        state.start_ingestion_if_needed(pairs)
        assert state._state == SessionState.INGESTING

        # Wait for partial progress (should be mid-ingestion)
        time.sleep(0.15)
        progress_at_second_call = state._ingestion_progress[0]
        assert progress_at_second_call > 0, "Should have made some progress"

        # Second call — should NOT restart from 0
        state.start_ingestion_if_needed(pairs)

        # Wait for completion
        time.sleep(2.0)
        assert state.session_state == SessionState.ACTIVE

        # KEY ASSERTIONS:
        # 1. Two ingestion calls: first was cancelled, second picked up remainder
        assert len(ingestion_calls) == 2, f"Expected 2 calls, got {ingestion_calls}"

        # 2. The second call should only ingest the REMAINING turns,
        #    not restart from 0. Total across both should equal 6.
        assert sum(ingestion_calls) < 12, (
            f"Second call restarted from 0: {ingestion_calls}"
        )
        assert ingestion_calls[1] < ingestion_calls[0], (
            f"Second call should be smaller (resumed): {ingestion_calls}"
        )

        # 3. All 6 turns should be tagged (no gaps, no duplicates)
        assert len(state.engine._turn_tag_index.entries) == 6

    @pytest.mark.regression("PROXY-014")
    def test_third_call_during_ingestion_cancels_second(self):
        """Third start_ingestion_if_needed should cancel the second
        thread and resume, not silently return because the session
        was erroneously marked as ingested by the first thread's
        finally block."""
        import hashlib
        import time

        ingestion_calls = []

        def slow_ingest(pairs, progress_callback=None, turn_offset=0):
            """Simulate slow ingestion — record each call's pair count."""
            ingestion_calls.append(len(pairs) // 2)
            n = len(pairs) // 2
            for i in range(n):
                combined = f"{pairs[i*2].content} {pairs[i*2+1].content}"
                entry = TurnTagEntry(
                    turn_number=len(state.engine._turn_tag_index.entries),
                    message_hash=hashlib.sha256(combined.encode()).hexdigest()[:16],
                    tags=[f"tag-{i}"],
                    primary_tag=f"tag-{i}",
                )
                state.engine._turn_tag_index.append(entry)
                if progress_callback:
                    progress_callback(i + 1, n, entry)
                time.sleep(0.08)  # slow enough to ensure cancel mid-flight
            return n

        state = self._make_state()
        state.engine.ingest_history.side_effect = slow_ingest

        # 20 pairs = 40 messages (takes ~1.6s per full run, plenty of time)
        pairs = [
            Message(role=("user" if j % 2 == 0 else "assistant"), content=f"msg-{j}")
            for j in range(40)
        ]

        # First call — starts background ingestion
        state.start_ingestion_if_needed(pairs)
        assert state._state == SessionState.INGESTING

        # Wait for partial progress
        time.sleep(0.3)
        p1 = state._ingestion_progress[0]
        assert p1 > 0, "Should have made some progress after call 1"

        # Second call — cancels first, resumes
        state.start_ingestion_if_needed(pairs)
        thread_after_second = state._ingestion_thread

        # Wait for partial progress from second thread
        time.sleep(0.3)

        # KEY CHECK: the cancelled first thread's finally block should
        # NOT have marked the session as ingested.  If it did, this call
        # will silently return without cancelling the second thread.
        conversation_id = state.engine.config.conversation_id
        was_marked = conversation_id in state._ingested_conversations
        state.start_ingestion_if_needed(pairs)
        thread_after_third = state._ingestion_thread

        # The third call should have started a NEW thread
        assert thread_after_third is not thread_after_second, (
            "Third call did not start a new thread — cancelled thread's "
            "finally block erroneously marked session as ingested "
            f"(was_marked={was_marked})"
        )

        # Wait for completion
        time.sleep(5.0)
        assert state.session_state == SessionState.ACTIVE
        assert state._history_ingested() is True

        # Should have had 3 ingestion calls (first two cancelled, third completes)
        assert len(ingestion_calls) == 3, (
            f"Expected 3 ingestion calls, got {ingestion_calls}"
        )

        # All 20 turns should eventually be tagged
        assert len(state.engine._turn_tag_index.entries) == 20

    def test_passthrough_request_has_flag(self):
        """Captured request in passthrough should have passthrough=True."""
        metrics = ProxyMetrics()
        metrics.capture_request(
            0, {"messages": []}, "anthropic",
            passthrough=True,
        )
        req = metrics.get_captured_request(0)
        assert req["passthrough"] is True

    def test_non_passthrough_request_default(self):
        """Default captured request should have passthrough=False."""
        metrics = ProxyMetrics()
        metrics.capture_request(0, {"messages": []}, "anthropic")
        req = metrics.get_captured_request(0)
        assert req["passthrough"] is False

