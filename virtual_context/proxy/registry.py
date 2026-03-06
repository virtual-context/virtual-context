"""Conversation registry for multi-conversation proxy routing.

Contains SessionRegistry — manages multiple concurrent ProxyState instances,
one per conversation, with fingerprint-based routing and persistence.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .formats import detect_format
from .metrics import ProxyMetrics
from .state import ProxyState

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)


class SessionRegistry:
    """Manages multiple concurrent ProxyState instances, one per conversation.

    Routing priority:
    1. Conversation marker (``<!-- vc:conversation=UUID -->``) in assistant messages
    2. Trailing fingerprint — hash of last N user messages (before current
       turn) in the request body, matched against in-memory or persisted
       fingerprints from previous conversations
    3. Fallback — claim unclaimed conversation or create new

    Trailing fingerprints survive client-side compaction (which rewrites
    early messages) because they sample from the tail of the history.

    Future: ``X-VC-Session`` request header overrides all (requires client changes).
    """

    _FINGERPRINT_SAMPLE_SIZE = 1  # last N user messages to hash

    def __init__(
        self,
        config_path: str | None,
        upstream: str,
        metrics: ProxyMetrics,
        *,
        store: "Store | None" = None,
    ) -> None:
        self._config_path = config_path
        self._upstream = upstream
        self._metrics = metrics
        self._conversations: dict[str, ProxyState] = {}
        self._fingerprints: dict[str, str] = {}  # fingerprint → conversation_id
        self._lock = threading.Lock()
        self._store = store  # for loading persisted fingerprints on restart

    @staticmethod
    def _msg_text(msg: dict) -> str:
        """Extract plain text from a message content (str or content blocks)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return ""

    @staticmethod
    def _compute_fingerprint(body: dict, offset: int = 0) -> str:
        """Trailing conversation fingerprint from the last N user messages.

        Delegates to the format-specific ``compute_fingerprint`` method
        which handles provider-specific message structures (Anthropic,
        OpenAI Chat, OpenAI Responses, Gemini).

        The ``offset`` parameter shifts the sampling window back from the
        tail.  Two conventions:

        - **offset=0** (store): hash the last S history messages.
          Used in ``catch_all`` to persist the fingerprint after each turn.
        - **offset=1** (match): hash S messages ending one position before
          the tail.  Used in ``get_or_create`` and restart matching.
        """
        fmt = detect_format(body)
        return fmt.compute_fingerprint(body, offset)

    def _match_persisted_fingerprint(self, body: dict) -> str | None:
        """Match inbound request against persisted conversation fingerprints.

        Compares the request's tail-1 fingerprint (offset=1) against stored
        tail fingerprints (offset=0).  The one-turn shift between the last
        save and the next inbound request is exactly bridged by offset=1.

        Returns the matched conversation_id, or None.
        """
        if not self._store:
            return None
        try:
            persisted = self._store.list_engine_state_fingerprints()
            if not isinstance(persisted, dict) or not persisted:
                return None
        except Exception:
            return None

        fp = self._compute_fingerprint(body, offset=1)
        if fp and fp in persisted:
            matched = persisted[fp]
            if isinstance(matched, str) and matched:
                logger.info(
                    "Persisted fingerprint match: fp=%s → conversation=%s",
                    fp[:8], matched[:12],
                )
                return matched
        return None

    def get_or_create(
        self,
        conversation_id: str | None,
        *,
        body: dict | None = None,
    ) -> tuple[ProxyState, bool]:
        """Look up or create a ProxyState for the given conversation ID.

        Returns (state, is_new).

        Routing priority: marker (conversation_id) > in-memory fingerprint >
        persisted fingerprint > claim unclaimed session > create new session.
        """
        # Fast path: conversation marker found and conversation already in memory
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id], False

        # Compute tail-1 fingerprint for matching (offset=1).
        # The incoming request's tail-1 matches the previous request's tail
        # because the conversation grew by exactly one turn.  catch_all
        # stores each request's tail (offset=0) in _fingerprints, so the
        # match here uses offset=1 to align with the stored value.
        fp_match = ""  # offset=1 — for matching against stored tail
        fp_store = ""  # offset=0 — saved in _fingerprints after session created
        if conversation_id is None and body is not None:
            fp_match = self._compute_fingerprint(body, offset=1)
            fp_store = self._compute_fingerprint(body)
            # Fast path: in-memory fingerprint match
            if fp_match and fp_match in self._fingerprints:
                matched_sid = self._fingerprints[fp_match]
                if matched_sid in self._conversations:
                    return self._conversations[matched_sid], False

        with self._lock:
            # Double-check after acquiring lock
            if conversation_id and conversation_id in self._conversations:
                return self._conversations[conversation_id], False

            if fp_match and fp_match in self._fingerprints:
                matched_sid = self._fingerprints[fp_match]
                if matched_sid in self._conversations:
                    return self._conversations[matched_sid], False

            # Check persisted fingerprints (survives proxy restart +
            # client-side compaction that destroys conversation markers)
            if conversation_id is None and body is not None:
                persisted_sid = self._match_persisted_fingerprint(body)
                if persisted_sid:
                    conversation_id = persisted_sid

            # No marker, no fingerprint match — claim an unclaimed conversation
            # if one exists.  This handles the startup case: create_app makes
            # a default conversation before any request arrives.  The first
            # conversation claims it; a second distinct conversation creates
            # a new one.
            if conversation_id is None:
                claimed_sids = set(self._fingerprints.values())
                for sid, st in self._conversations.items():
                    if sid not in claimed_sids:
                        if fp_store:
                            self._fingerprints[fp_store] = sid
                        logger.info(
                            "Conversation claimed: %s (fp=%s, total=%d)",
                            sid[:12],
                            fp_store[:8] if fp_store else "none",
                            len(self._conversations),
                        )
                        return st, False

            # Create a new engine instance.
            # Late import: tests patch VirtualContextEngine on the server
            # module (``virtual_context.proxy.server.VirtualContextEngine``),
            # so we look it up through that namespace to stay patchable.
            from . import server as _srv  # noqa: avoid circular at module level
            engine = _srv.VirtualContextEngine(config_path=self._config_path)

            if conversation_id:
                # Override the auto-generated conversation_id so load_engine_state
                # can find the persisted state for this session.
                engine.config.conversation_id = conversation_id
                # Trigger state reload (engine.__init__ already called
                # _load_persisted_state but with the wrong conversation_id).
                engine._load_persisted_state()
                engine._bootstrap_vocabulary()

            actual_id = engine.config.conversation_id
            state = ProxyState(
                engine, metrics=self._metrics, upstream=self._upstream,
            )
            self._conversations[actual_id] = state

            # Record fingerprint → conversation mapping (offset=0 = tail)
            if fp_store:
                self._fingerprints[fp_store] = actual_id

            logger.info(
                "Conversation %s: %s (fp=%s, total=%d)",
                "resumed" if conversation_id else "created",
                actual_id[:12],
                fp_store[:8] if fp_store else "none",
                len(self._conversations),
            )
            return state, True

    @property
    def conversation_count(self) -> int:
        return len(self._conversations)

    def shutdown_all(self) -> None:
        """Shut down all session states."""
        for state in self._conversations.values():
            state.shutdown()
        self._conversations.clear()
