"""Conversation registry for multi-conversation proxy routing.

Contains SessionRegistry — manages multiple concurrent ProxyState instances,
one per conversation, with fingerprint-based routing and persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import TYPE_CHECKING

from .metrics import ProxyMetrics
from .state import ProxyState

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)


class SessionRegistry:
    """Manages multiple concurrent ProxyState instances, one per conversation.

    Routing priority:
    1. System prompt hash — stable across turns, survives client compaction,
       unique per chat (OpenClaw embeds ``chat_id`` in the system prompt)
    2. Last-message hash — hash of the last user message from the previous
       request matches against the second-to-last user message in the
       current request (fallback when system prompt is absent)
    3. Claim unclaimed session / create new
    """

    def __init__(
        self,
        config_path: str | None,
        upstream: str,
        metrics: ProxyMetrics,
        *,
        store: "Store | None" = None,
        session_cache=None,
        embedding_provider=None,
    ) -> None:
        self._config_path = config_path
        self._upstream = upstream
        self._metrics = metrics
        self._conversations: dict[str, ProxyState] = {}
        self._sys_hashes: dict[str, str] = {}  # system_prompt_hash → conversation_id
        self._chat_ids: dict[str, str] = {}  # chat_id_value → conversation_id
        # Last-message fingerprint: hash of the last user message from the
        # most recent request for each conversation.  The next request's
        # second-to-last user message should match this.
        self._last_msg_hashes: dict[str, str] = {}  # hash → conversation_id
        self._lock = threading.Lock()
        self._store = store  # for loading persisted fingerprints on restart
        self._session_cache = session_cache
        self._embedding_provider = embedding_provider

    @staticmethod
    def _compute_system_hash(body: dict) -> str:
        """Hash the system prompt to identify the conversation.

        The system prompt contains per-chat metadata (e.g. OpenClaw embeds
        ``chat_id``) and is stable across turns — never compacted by the
        client.  This makes it the most reliable routing signal for a
        transparent proxy.

        Handles both Anthropic (``system`` key, str or list of blocks) and
        OpenAI (first message with ``role: "system"``).
        """
        text = ""

        # Anthropic: top-level "system" key
        system = body.get("system")
        if system is not None:
            if isinstance(system, str):
                text = system
            elif isinstance(system, list):
                text = " ".join(
                    b.get("text", "") for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        # OpenAI: first message with role "system" or "developer"
        if not text:
            for msg in body.get("messages", []):
                role = msg.get("role", "")
                if role in ("system", "developer"):
                    c = msg.get("content", "")
                    text = c if isinstance(c, str) else " ".join(
                        b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text"
                    ) if isinstance(c, list) else ""
                    break

        if not text:
            return ""

        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_user_message(body: dict, position: int) -> str:
        """Hash a user message at a given position from the end.

        position=0 → last user message (current turn)
        position=1 → second-to-last user message (previous turn)

        After each request we store hash(position=0).  On the next
        request we compute hash(position=1) and match — because what
        was the last message last time is now the second-to-last.
        """
        msgs = body.get("messages", [])
        user_msgs = [
            m for m in msgs
            if m.get("role") == "user"
        ]
        idx = len(user_msgs) - 1 - position
        if idx < 0:
            return ""

        content = user_msgs[idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not content:
            return ""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    _SYS_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

    @staticmethod
    def _extract_chat_id(body: dict) -> str:
        """Extract ``chat_id`` from JSON code blocks in the system prompt.

        Returns the chat_id string, or empty string if not found.
        """
        text = ""

        system = body.get("system")
        if system is not None:
            if isinstance(system, str):
                text = system
            elif isinstance(system, list):
                text = "\n".join(
                    b.get("text", "") for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        if not text:
            for msg in body.get("messages", []):
                role = msg.get("role", "")
                if role in ("system", "developer"):
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = "\n".join(
                            b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    break

        if not text:
            return ""

        for m in SessionRegistry._SYS_JSON_BLOCK_RE.finditer(text):
            try:
                parsed = json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("chat_id"), str):
                return parsed["chat_id"]
        return ""

    def update_last_message_hash(self, body: dict, conversation_id: str) -> None:
        """Store the hash of the current last user message for this conversation.

        Called after routing succeeds so the next request can match against it.
        """
        h = self._hash_user_message(body, position=0)
        if h:
            self._last_msg_hashes[h] = conversation_id

    def get_or_create(
        self,
        conversation_id: str | None,
        *,
        body: dict | None = None,
    ) -> tuple[ProxyState, bool]:
        """Look up or create a ProxyState for the given conversation ID.

        Returns (state, is_new).

        Routing priority:
        1. System prompt hash (stable per-chat identifier)
        2. Last-message hash (previous turn's last msg = this turn's second-to-last)
        3. Claim unclaimed session (first request after startup)
        4. Create new session
        """
        # --- 0. Explicit conversation marker (highest priority) ---
        if conversation_id:
            for sid, st in self._conversations.items():
                if st.engine.config.conversation_id == conversation_id:
                    if body is not None:
                        self.update_last_message_hash(body, sid)
                    return st, False

        sys_hash = ""
        msg_hash = ""
        chat_id = ""
        if body is not None:
            chat_id = self._extract_chat_id(body)
            sys_hash = self._compute_system_hash(body)
            msg_hash = self._hash_user_message(body, position=1)

        # --- 1. chat_id from system prompt (stable per-chat) ---
        if chat_id and chat_id in self._chat_ids:
            target_sid = self._chat_ids[chat_id]
            if target_sid in self._conversations:
                self.update_last_message_hash(body, target_sid)
                return self._conversations[target_sid], False

        # --- 2. System prompt hash (primary) ---
        if sys_hash and sys_hash in self._sys_hashes:
            matched_sid = self._sys_hashes[sys_hash]
            if matched_sid in self._conversations:
                self.update_last_message_hash(body, matched_sid)
                return self._conversations[matched_sid], False

        # --- 3. Last-message hash (secondary) ---
        if msg_hash and msg_hash in self._last_msg_hashes:
            matched_sid = self._last_msg_hashes[msg_hash]
            if matched_sid in self._conversations:
                self.update_last_message_hash(body, matched_sid)
                return self._conversations[matched_sid], False

        with self._lock:
            # Double-check after acquiring lock
            if chat_id and chat_id in self._chat_ids:
                target_sid = self._chat_ids[chat_id]
                if target_sid in self._conversations:
                    self.update_last_message_hash(body, target_sid)
                    return self._conversations[target_sid], False

            if sys_hash and sys_hash in self._sys_hashes:
                matched_sid = self._sys_hashes[sys_hash]
                if matched_sid in self._conversations:
                    self.update_last_message_hash(body, matched_sid)
                    return self._conversations[matched_sid], False

            if msg_hash and msg_hash in self._last_msg_hashes:
                matched_sid = self._last_msg_hashes[msg_hash]
                if matched_sid in self._conversations:
                    self.update_last_message_hash(body, matched_sid)
                    return self._conversations[matched_sid], False

            # --- 4. Claim unclaimed session ---
            claimed_sids = set(self._sys_hashes.values()) | set(self._last_msg_hashes.values()) | set(self._chat_ids.values())
            for sid, st in self._conversations.items():
                if sid not in claimed_sids:
                    if chat_id:
                        self._chat_ids[chat_id] = sid
                    if sys_hash:
                        self._sys_hashes[sys_hash] = sid
                    if body is not None:
                        self.update_last_message_hash(body, sid)
                    logger.info(
                        "Conversation claimed: %s (sys=%s, total=%d)",
                        sid[:12],
                        sys_hash[:8] if sys_hash else "none",
                        len(self._conversations),
                    )
                    return st, False

            # --- 5. Create new session ---
            from . import server as _srv

            if conversation_id:
                # Marker path: set conv_id BEFORE construction so Redis loads the right key
                from ..config import load_config
                _cfg = load_config(self._config_path)
                _cfg.conversation_id = conversation_id
                engine = _srv.VirtualContextEngine(
                    config=_cfg, session_cache=self._session_cache,
                    embedding_provider=self._embedding_provider,
                )
                # No need for post-construction rebind — engine has the right ID from start
            else:
                engine = _srv.VirtualContextEngine(
                    config_path=self._config_path, session_cache=self._session_cache,
                    embedding_provider=self._embedding_provider,
                )

            actual_id = engine.config.conversation_id
            state = ProxyState(
                engine, metrics=self._metrics, upstream=self._upstream,
            )
            self._conversations[actual_id] = state

            if chat_id:
                self._chat_ids[chat_id] = actual_id
            if sys_hash:
                self._sys_hashes[sys_hash] = actual_id
            if body is not None:
                self.update_last_message_hash(body, actual_id)

            logger.info(
                "Conversation created: %s (sys=%s, total=%d)",
                actual_id[:12],
                sys_hash[:8] if sys_hash else "none",
                len(self._conversations),
            )
            return state, True

    @property
    def conversation_count(self) -> int:
        return len(self._conversations)

    def get_state(self, conversation_id: str) -> ProxyState | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def first_state(self) -> ProxyState | None:
        with self._lock:
            return next(iter(self._conversations.values()), None)

    def remove_conversation(self, conversation_id: str) -> ProxyState | None:
        with self._lock:
            state = self._conversations.pop(conversation_id, None)
            self._sys_hashes = {
                key: value for key, value in self._sys_hashes.items()
                if value != conversation_id
            }
            self._chat_ids = {
                key: value for key, value in self._chat_ids.items()
                if value != conversation_id
            }
            self._last_msg_hashes = {
                key: value for key, value in self._last_msg_hashes.items()
                if value != conversation_id
            }
            return state

    def shutdown_all(self) -> None:
        with self._lock:
            states = list(self._conversations.values())
            self._conversations.clear()
            self._sys_hashes.clear()
            self._chat_ids.clear()
            self._last_msg_hashes.clear()
        for state in states:
            state.shutdown()
