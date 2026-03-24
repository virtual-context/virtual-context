"""Optional Redis-backed session cache for fast engine state restore."""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_lib = None  # type: ignore
    _REDIS_AVAILABLE = False


class RedisSessionCache:
    """Write-through Redis cache for conversation state snapshots.

    Single key per conversation: vc:{conv_id}:snapshot
    Contains atomic snapshot of history + TurnTagIndex + engine state.

    All operations retry 3x with backoff. On failure, enters degraded mode
    (store-only). Recovers automatically when Redis comes back.
    """

    _RETRY_BACKOFFS_MS = [50, 100, 200]
    _RECOVERY_CHECK_INTERVAL = 60.0  # seconds

    def __init__(self, redis_url: str, history_cap: int = 600) -> None:
        self._url = redis_url
        self._history_cap = history_cap
        self._redis = None
        self._redis_available = False
        self._degraded = False
        self._last_health_check = 0.0
        self._degraded_warned = False

        if not redis_url:
            return
        if not _REDIS_AVAILABLE:
            logger.warning("redis_url configured but redis package not installed — cache disabled")
            return

        try:
            self._redis = _redis_lib.Redis.from_url(redis_url, decode_responses=False)
            self._redis.ping()
            self._redis_available = True
            logger.info("Redis session cache connected (%s)", redis_url)
        except Exception as e:
            logger.warning("Redis connection failed (%s): %s — cache disabled", redis_url, e)

    @property
    def history_cap(self) -> int:
        return self._history_cap

    def _key(self, conv_id: str) -> str:
        return f"vc:{conv_id}:snapshot"

    def _retry(self, fn, *args):
        """Execute fn with 3 retries + exponential backoff. Raises on all failures."""
        last_err = None
        for attempt, backoff_ms in enumerate(self._RETRY_BACKOFFS_MS):
            try:
                return fn(*args)
            except Exception as e:
                last_err = e
                if attempt < len(self._RETRY_BACKOFFS_MS) - 1:
                    time.sleep(backoff_ms / 1000)
        raise last_err  # type: ignore

    def is_available(self) -> bool:
        return self._redis_available and not self._degraded

    def _maybe_recover(self) -> None:
        """Check if Redis is back. Called periodically when degraded."""
        now = time.time()
        if now - self._last_health_check < self._RECOVERY_CHECK_INTERVAL:
            return
        self._last_health_check = now
        if not self._redis:
            return
        try:
            self._redis.ping()
            self._degraded = False
            self._degraded_warned = False
            logger.info("Redis recovered — session cache re-enabled")
        except Exception:
            pass

    def save_snapshot(self, conv_id: str, snapshot: dict) -> None:
        if not self._redis_available:
            return
        if self._degraded:
            self._maybe_recover()
            if self._degraded:
                return
        try:
            data = json.dumps(snapshot, default=str).encode("utf-8")
            self._retry(self._redis.set, self._key(conv_id), data)
        except Exception as e:
            if not self._degraded_warned:
                logger.warning("Redis write failed after retries — entering degraded mode: %s", e)
                self._degraded_warned = True
            self._degraded = True

    def load_snapshot(self, conv_id: str) -> dict | None:
        if not self._redis_available:
            return None
        if self._degraded:
            self._maybe_recover()
            if self._degraded:
                return None
        try:
            raw = self._retry(self._redis.get, self._key(conv_id))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("Redis read failed: %s", e)
            return None

    def delete_conversation(self, conv_id: str) -> None:
        if not self._redis_available:
            return
        try:
            self._redis.delete(self._key(conv_id))
        except Exception:
            pass  # best effort

    def flush_snapshot_background(self, conv_id: str, snapshot: dict) -> None:
        """Write snapshot in a background thread (used for recovery flush)."""
        import threading
        def _flush():
            self.save_snapshot(conv_id, snapshot)
            logger.info("Redis recovery flush complete: conv=%s version=%s",
                        conv_id[:12], snapshot.get("version", "?"))
        threading.Thread(target=_flush, daemon=True).start()
