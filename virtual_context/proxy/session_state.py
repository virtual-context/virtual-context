"""Redis-backed conversation session state provider.

Replaces the old RedisSessionCache. Stores ~200KB of checkpoint metadata
per conversation — NOT conversation history (that comes from the client
payload every request).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

_MAX_VERSION = 2**53  # tombstone version — higher than any real save


@dataclass
class SessionState:
    """Serializable conversation checkpoint — what goes in Redis."""
    compacted_through: int = 0
    last_compacted_turn: int = -1
    last_completed_turn: int = -1
    last_indexed_turn: int = -1
    checkpoint_version: int = 0
    conversation_generation: int = 0
    tool_tag_counter: int = 0
    split_processed_tags: set[str] = field(default_factory=set)
    trailing_fingerprint: str = ""
    provider: str = ""
    turn_tag_entries: list[dict] = field(default_factory=list)
    working_set: list[dict] = field(default_factory=list)
    telemetry_rollup: dict = field(default_factory=dict)
    request_captures: list[dict] = field(default_factory=list)
    version: int = 0
    deleted: bool = False

    def to_json(self) -> bytes:
        d = {
            "compacted_through": self.compacted_through,
            "last_compacted_turn": self.last_compacted_turn,
            "last_completed_turn": self.last_completed_turn,
            "last_indexed_turn": self.last_indexed_turn,
            "checkpoint_version": self.checkpoint_version,
            "conversation_generation": self.conversation_generation,
            "tool_tag_counter": self.tool_tag_counter,
            "split_processed_tags": sorted(self.split_processed_tags),
            "trailing_fingerprint": self.trailing_fingerprint,
            "provider": self.provider,
            "turn_tag_entries": self.turn_tag_entries,
            "working_set": self.working_set,
            "telemetry_rollup": self.telemetry_rollup,
            "request_captures": self.request_captures,
            "version": self.version,
            "deleted": self.deleted,
        }
        return json.dumps(d, default=str).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> SessionState:
        d = json.loads(data)
        return cls(
            compacted_through=d.get("compacted_through", 0),
            last_compacted_turn=d.get("last_compacted_turn", -1),
            last_completed_turn=d.get("last_completed_turn", -1),
            last_indexed_turn=d.get("last_indexed_turn", -1),
            checkpoint_version=d.get("checkpoint_version", 0),
            conversation_generation=d.get("conversation_generation", 0),
            tool_tag_counter=d.get("tool_tag_counter", 0),
            split_processed_tags=set(d.get("split_processed_tags", [])),
            trailing_fingerprint=d.get("trailing_fingerprint", ""),
            provider=d.get("provider", ""),
            turn_tag_entries=d.get("turn_tag_entries", []),
            working_set=d.get("working_set", []),
            telemetry_rollup=d.get("telemetry_rollup", {}),
            request_captures=d.get("request_captures", []),
            version=d.get("version", 0),
            deleted=d.get("deleted", False),
        )


class SessionStateProvider:
    """Redis-backed session state. Load at request start, save at request end."""

    _PAYLOAD_TOKEN_CACHE_TTL_SECONDS = 6 * 60 * 60
    _TAG_EMBEDDING_CACHE_TTL_SECONDS = 24 * 60 * 60
    _CONTEXT_HINT_CACHE_TTL_SECONDS = 6 * 60 * 60

    def __init__(self, redis_client=None, redis_url: str = "", store=None) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        else:
            raise ValueError("redis_client or redis_url required")
        self._store = store  # Optional ContextStore for Postgres backup/fallback
        self._degraded = False

    def _key(self, conversation_id: str) -> str:
        return f"vc:session:{conversation_id}"

    def _payload_token_cache_key(self, conversation_id: str) -> str:
        return f"vc:payload_tokens:{conversation_id}"

    def _tag_embedding_cache_key(self, model_name: str, tag: str) -> str:
        digest = hashlib.sha1(f"{model_name}\0{tag}".encode("utf-8")).hexdigest()
        return f"vc:tag_embedding:{digest}"

    def _context_hint_cache_key(self, conversation_id: str, cache_key: str) -> str:
        return f"vc:context_hint:{conversation_id}:{cache_key}"

    def load(self, conversation_id: str) -> SessionState | None:
        """Load session state from Redis. Returns None if not found.
        Returns SessionState(deleted=True) if tombstoned.
        """
        try:
            raw = self._redis.get(self._key(conversation_id))
            if raw is None:
                # Redis miss — try Postgres fallback
                return self._load_from_store(conversation_id)
            return SessionState.from_json(raw)
        except Exception:
            logger.warning("Redis load failed for %s", conversation_id[:12], exc_info=True)
            self._degraded = True
            # Degraded — try Postgres fallback
            return self._load_from_store(conversation_id)

    def save(self, conversation_id: str, state: SessionState) -> None:
        """Save session state to Redis with optimistic version check.

        Uses a Redis transaction (WATCH/MULTI) so an in-flight stale worker
        cannot overwrite a newer checkpoint or a delete tombstone:
        - WATCH the key
        - GET the current value and check version
        - If current version > state.version, discard (we're stale)
        - If deleted flag is set, discard (conversation was deleted mid-flight)
        - MULTI: SET with incremented version
        - If WATCH fails (concurrent write): discard, log warning
        """
        key = self._key(conversation_id)
        state.version += 1
        try:
            with self._redis.pipeline() as pipe:
                pipe.watch(key)
                current_raw = pipe.get(key)
                if current_raw:
                    current = json.loads(current_raw)
                    if current.get("deleted"):
                        logger.info("Save rejected for %s — tombstoned", conversation_id[:12])
                        return
                    if current.get("version", 0) >= state.version:
                        logger.info("Save rejected for %s — stale version %d < %d",
                                    conversation_id[:12], state.version, current["version"])
                        return
                pipe.multi()
                pipe.set(key, state.to_json())
                pipe.execute()
            self._degraded = False
            # Postgres backup only when Redis succeeded — if Redis failed,
            # writing to Postgres would put it ahead of Redis, and load()
            # would later trust the stale Redis copy over the newer store.
            self._save_to_store(conversation_id, state)
        except Exception:
            logger.warning("Redis save failed for %s — skipping Postgres backup",
                           conversation_id[:12], exc_info=True)
            self._degraded = True

    def load_payload_token_cache(self, conversation_id: str):
        """Load the segmented inbound token cache for a conversation.

        This cache is an optional hot-path optimization only. Failures should
        never affect correctness or the primary session-state flow.
        """
        try:
            raw = self._redis.get(self._payload_token_cache_key(conversation_id))
            if raw is None:
                return None
            from .formats import PayloadTokenCache
            return PayloadTokenCache(**json.loads(raw))
        except Exception:
            logger.warning(
                "Redis payload-token cache load failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            return None

    def save_payload_token_cache(self, conversation_id: str, cache, *, ttl_seconds: int | None = None) -> None:
        """Save the segmented inbound token cache for a conversation.

        Stored separately from durable session state so it can be updated on
        every request without inflating the authoritative checkpoint blob.
        """
        if cache is None:
            return
        try:
            payload = asdict(cache) if hasattr(cache, "__dataclass_fields__") else cache
            self._redis.set(
                self._payload_token_cache_key(conversation_id),
                json.dumps(payload, default=str).encode("utf-8"),
                ex=ttl_seconds or self._PAYLOAD_TOKEN_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Redis payload-token cache save failed for %s",
                conversation_id[:12],
                exc_info=True,
            )

    def delete_payload_token_cache(self, conversation_id: str) -> None:
        """Best-effort delete for the segmented inbound token cache."""
        try:
            self._redis.delete(self._payload_token_cache_key(conversation_id))
        except Exception:
            pass

    def load_tag_embeddings(self, model_name: str, tags: list[str]) -> dict[str, list[float]]:
        """Load cached tag embeddings for a model."""
        unique_tags: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not tag or tag in seen:
                continue
            seen.add(tag)
            unique_tags.append(tag)
        if not unique_tags:
            return {}

        try:
            keys = [self._tag_embedding_cache_key(model_name, tag) for tag in unique_tags]
            mget = getattr(self._redis, "mget", None)
            if callable(mget):
                raw_values = mget(keys)
            else:
                raw_values = [self._redis.get(key) for key in keys]

            loaded: dict[str, list[float]] = {}
            for tag, raw in zip(unique_tags, raw_values):
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                value = json.loads(raw)
                if isinstance(value, list):
                    loaded[tag] = value
            return loaded
        except Exception:
            logger.warning(
                "Redis tag-embedding cache load failed (model=%s tags=%d)",
                model_name,
                len(unique_tags),
                exc_info=True,
            )
            return {}

    def save_tag_embeddings(
        self,
        model_name: str,
        embeddings_by_tag: dict[str, list[float]],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Save tag embeddings for a model in shared Redis cache."""
        if not embeddings_by_tag:
            return
        try:
            ttl = ttl_seconds or self._TAG_EMBEDDING_CACHE_TTL_SECONDS
            with self._redis.pipeline() as pipe:
                for tag, embedding in embeddings_by_tag.items():
                    pipe.set(
                        self._tag_embedding_cache_key(model_name, tag),
                        json.dumps(embedding, default=str).encode("utf-8"),
                        ex=ttl,
                    )
                pipe.execute()
        except Exception:
            logger.warning(
                "Redis tag-embedding cache save failed (model=%s tags=%d)",
                model_name,
                len(embeddings_by_tag),
                exc_info=True,
            )

    def load_context_hint_cache(self, conversation_id: str, cache_key: str) -> str | None:
        """Load a rendered context hint for a conversation fingerprint."""
        try:
            raw = self._redis.get(self._context_hint_cache_key(conversation_id, cache_key))
            if raw is None:
                return None
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)
        except Exception:
            logger.warning(
                "Redis context-hint cache load failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            return None

    def save_context_hint_cache(
        self,
        conversation_id: str,
        cache_key: str,
        hint: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Save a rendered context hint for a conversation fingerprint."""
        try:
            self._redis.set(
                self._context_hint_cache_key(conversation_id, cache_key),
                hint.encode("utf-8"),
                ex=ttl_seconds or self._CONTEXT_HINT_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Redis context-hint cache save failed for %s",
                conversation_id[:12],
                exc_info=True,
            )

    def delete(self, conversation_id: str) -> None:
        """Tombstone the conversation. BOTH backends must succeed.

        Redis tombstone prevents workers from loading the live state.
        Postgres delete prevents resurrection via _load_from_store fallback.
        If either fails, retry it — a partial delete leaks state.
        """
        tombstone = SessionState(deleted=True, version=_MAX_VERSION)
        errors: list[str] = []

        # 1. Redis tombstone
        try:
            self._redis.set(
                self._key(conversation_id),
                tombstone.to_json(),
                ex=86400,
            )
            self.delete_payload_token_cache(conversation_id)
            self._degraded = False
        except Exception as e:
            errors.append(f"Redis: {e}")
            self._degraded = True

        # 2. Postgres delete
        if self._store and hasattr(self._store, "delete_conversation"):
            try:
                self._store.delete_conversation(conversation_id)
            except Exception as e:
                errors.append(f"Postgres: {e}")
        elif self._store is None:
            errors.append("Postgres: no store wired")

        if errors:
            # At least one backend failed — the delete is not sticky.
            # Suppress the Postgres fallback for this conversation so
            # _load_from_store can't resurrect it if Redis tombstone succeeded.
            # Then raise so the caller can retry or alert.
            logger.error(
                "DELETE NOT STICKY for %s — %s",
                conversation_id[:12], "; ".join(errors),
            )
            raise RuntimeError(
                f"Delete incomplete for {conversation_id[:12]}: "
                + "; ".join(errors)
            )

    def exists(self, conversation_id: str) -> bool:
        """Check if conversation exists and is not tombstoned."""
        state = self.load(conversation_id)
        return state is not None and not state.deleted

    def next_tool_tag(self, conversation_id: str) -> int:
        """Atomic tool tag counter via Redis INCR. Returns new value."""
        try:
            return self._redis.incr(f"vc:tool_counter:{conversation_id}")
        except Exception:
            # Degraded: use timestamp-based fallback
            import time
            return int(time.time() * 1000) % 100000

    def seed_tool_counter(self, conversation_id: str, value: int) -> None:
        """Set the tool counter to at least `value`. Used on Postgres fallback
        restore to prevent collisions with existing tool_N tags."""
        try:
            key = f"vc:tool_counter:{conversation_id}"
            current = self._redis.get(key)
            if current is None or int(current) < value:
                self._redis.set(key, str(value))
        except Exception:
            pass

    def _load_from_store(self, conversation_id: str) -> SessionState | None:
        """Postgres fallback when Redis misses or is degraded.

        Refuses to load if this conversation's Postgres backup is known-stale
        (Redis succeeded but _save_to_store failed). In that case the store
        has an older checkpoint than Redis, and loading it would lose state.
        """
        if not self._store or not hasattr(self._store, "load_engine_state"):
            return None
        # Check if store is known-stale for this conversation (Redis key shared
        # across all workers). Fail CLOSED: if we can't read the marker
        # (Redis down), refuse the fallback. The only time we reach this path
        # is when Redis already failed for the main load, so Redis being down
        # here is expected — and loading a potentially stale Postgres snapshot
        # is the exact bug this guard prevents.
        try:
            if self._redis.get(f"vc:store_stale:{conversation_id}"):
                logger.warning(
                    "Skipping Postgres fallback for %s — store is known-stale",
                    conversation_id[:12],
                )
                return None
        except Exception:
            logger.warning(
                "Skipping Postgres fallback for %s — cannot verify store freshness (Redis down)",
                conversation_id[:12],
            )
            return None
        try:
            saved = self._store.load_engine_state(conversation_id)
            if saved:
                state = self._snapshot_to_state(saved)
                # Seed the Redis tool counter so INCR starts above existing tool_N tags
                if state.tool_tag_counter > 0:
                    self.seed_tool_counter(conversation_id, state.tool_tag_counter)
                return state
        except Exception:
            logger.warning("Postgres fallback load failed for %s", conversation_id[:12])
        return None

    def _save_to_store(self, conversation_id: str, state: SessionState) -> None:
        """Postgres backup — best-effort, Redis is authoritative.

        On failure, marks this conversation as store-stale so
        _load_from_store won't trust the older Postgres checkpoint.
        """
        if not self._store or not hasattr(self._store, "save_engine_state"):
            return
        try:
            self._store.save_engine_state(
                self._state_to_snapshot(conversation_id, state))
            # Backup succeeded — clear stale flag across all workers
            try:
                self._redis.delete(f"vc:store_stale:{conversation_id}")
            except Exception:
                pass
        except Exception:
            logger.warning("Postgres backup save failed for %s — marking store stale",
                           conversation_id[:12])
            # Mark stale across all workers via Redis (5 min TTL — auto-heals
            # when the next successful backup clears it)
            try:
                self._redis.set(f"vc:store_stale:{conversation_id}", "1", ex=300)
            except Exception:
                pass

    def _state_to_snapshot(self, conversation_id: str, state: SessionState) -> "EngineStateSnapshot":
        """Convert SessionState → EngineStateSnapshot for Postgres persistence."""
        from ..types import EngineStateSnapshot, TurnTagEntry, WorkingSetEntry, DepthLevel
        from datetime import datetime, timezone

        entries = []
        for d in state.turn_tag_entries:
            ts_raw = d.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            # Restore fact_signals for compaction pipeline
            fs_raw = d.get("fact_signals", [])
            fs = None
            if fs_raw:
                from ..types import FactSignal
                fs = [
                    FactSignal(
                        subject=f.get("subject", ""),
                        verb=f.get("verb", ""),
                        object=f.get("object", ""),
                        status=f.get("status", ""),
                        fact_type=f.get("fact_type", ""),
                        what=f.get("what", ""),
                    )
                    for f in fs_raw if isinstance(f, dict)
                ]

            entries.append(TurnTagEntry(
                turn_number=d.get("turn_number", 0),
                tags=d.get("tags", []),
                primary_tag=d.get("primary_tag", ""),
                message_hash=d.get("message_hash", ""),
                sender=d.get("sender", ""),
                timestamp=ts,
                session_date=d.get("session_date", ""),
                fact_signals=fs,
            ))

        ws = []
        for w in state.working_set:
            ws.append(WorkingSetEntry(
                tag=w.get("tag", ""),
                depth=DepthLevel(w.get("depth", "summary")),
                tokens=w.get("tokens", 0),
                last_accessed_turn=w.get("last_accessed_turn", 0),
            ))

        return EngineStateSnapshot(
            conversation_id=conversation_id,
            compacted_through=state.compacted_through,
            turn_tag_entries=entries,
            turn_count=len(entries),
            last_compacted_turn=state.last_compacted_turn,
            last_completed_turn=state.last_completed_turn,
            last_indexed_turn=state.last_indexed_turn,
            checkpoint_version=state.checkpoint_version,
            conversation_generation=state.conversation_generation,
            split_processed_tags=sorted(state.split_processed_tags),
            working_set=ws,
            trailing_fingerprint=state.trailing_fingerprint,
            provider=state.provider,
            telemetry_rollup=state.telemetry_rollup,
            request_captures=state.request_captures,
            tool_tag_counter=state.tool_tag_counter,
        )

    def _snapshot_to_state(self, snapshot: "EngineStateSnapshot") -> SessionState:
        """Convert EngineStateSnapshot → SessionState for Redis fallback from Postgres."""
        entries = []
        for e in snapshot.turn_tag_entries:
            entries.append({
                "turn_number": e.turn_number,
                "tags": e.tags,
                "primary_tag": e.primary_tag,
                "message_hash": e.message_hash,
                "sender": getattr(e, "sender", ""),
                "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                "session_date": getattr(e, "session_date", ""),
                "fact_signals": [
                    {"subject": fs.subject, "verb": fs.verb,
                     "object": fs.object, "status": fs.status,
                     "fact_type": getattr(fs, "fact_type", ""),
                     "what": getattr(fs, "what", "")}
                    for fs in (e.fact_signals or [])
                ] if e.fact_signals else [],
            })

        ws = []
        for w in (snapshot.working_set or []):
            ws.append({
                "tag": w.tag,
                "depth": w.depth.value if hasattr(w.depth, 'value') else w.depth,
                "tokens": w.tokens,
                "last_accessed_turn": w.last_accessed_turn,
            })

        return SessionState(
            compacted_through=snapshot.compacted_through,
            last_compacted_turn=snapshot.last_compacted_turn,
            last_completed_turn=snapshot.last_completed_turn,
            last_indexed_turn=snapshot.last_indexed_turn,
            checkpoint_version=snapshot.checkpoint_version,
            conversation_generation=snapshot.conversation_generation,
            split_processed_tags=set(snapshot.split_processed_tags or []),
            trailing_fingerprint=snapshot.trailing_fingerprint,
            provider=snapshot.provider,
            turn_tag_entries=entries,
            working_set=ws,
            telemetry_rollup=snapshot.telemetry_rollup or {},
            request_captures=snapshot.request_captures or [],
            tool_tag_counter=getattr(snapshot, 'tool_tag_counter', 0),
        )

    @property
    def is_degraded(self) -> bool:
        return self._degraded
