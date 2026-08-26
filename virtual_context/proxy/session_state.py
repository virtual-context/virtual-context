"""Redis-backed conversation session state provider.

Replaces the old RedisSessionCache. Stores ~200KB of checkpoint metadata
per conversation — NOT conversation history (that comes from the client
payload every request).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..types import TagStats

logger = logging.getLogger(__name__)

_MAX_VERSION = 2**53  # tombstone version — higher than any real save


@dataclass
class SessionState:
    """Serializable conversation checkpoint — what goes in Redis."""
    compacted_prefix_messages: int = 0
    flushed_prefix_messages: int = 0
    flushed_prefix_messages_present: bool = True
    last_request_time: float = 0.0
    last_compacted_turn: int = -1
    last_completed_turn: int = -1
    last_indexed_turn: int = -1
    checkpoint_version: int = 0
    conversation_generation: int = 0
    tool_tag_counter: int = 0
    split_processed_tags: set[str] = field(default_factory=set)
    trailing_fingerprint: str = ""
    provider: str = ""
    session_state: str = ""
    live_turn_count: int = 0
    history_message_count: int = 0
    ingestion_done: int = 0
    ingestion_total: int = 0
    last_payload_kb: float = 0.0
    last_payload_tokens: int = 0
    raw_payload_entry_count: int = 0
    ingestible_entry_count: int = 0
    skipped_payload_entry_count: int = 0
    turn_tag_entries: list[dict] = field(default_factory=list)
    working_set: list[dict] = field(default_factory=list)
    telemetry_rollup: dict = field(default_factory=dict)
    request_captures: list[dict] = field(default_factory=list)
    version: int = 0
    deleted: bool = False

    def to_json(self) -> bytes:
        d = {
            "compacted_prefix_messages": self.compacted_prefix_messages,
            "flushed_prefix_messages": self.flushed_prefix_messages,
            "last_request_time": self.last_request_time,
            "last_compacted_turn": self.last_compacted_turn,
            "last_completed_turn": self.last_completed_turn,
            "last_indexed_turn": self.last_indexed_turn,
            "checkpoint_version": self.checkpoint_version,
            "conversation_generation": self.conversation_generation,
            "tool_tag_counter": self.tool_tag_counter,
            "split_processed_tags": sorted(self.split_processed_tags),
            "trailing_fingerprint": self.trailing_fingerprint,
            "provider": self.provider,
            "session_state": self.session_state,
            "live_turn_count": self.live_turn_count,
            "history_message_count": self.history_message_count,
            "ingestion_done": self.ingestion_done,
            "ingestion_total": self.ingestion_total,
            "last_payload_kb": self.last_payload_kb,
            "last_payload_tokens": self.last_payload_tokens,
            "raw_payload_entry_count": self.raw_payload_entry_count,
            "ingestible_entry_count": self.ingestible_entry_count,
            "skipped_payload_entry_count": self.skipped_payload_entry_count,
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
            compacted_prefix_messages=d.get("compacted_prefix_messages", 0),
            flushed_prefix_messages=d.get("flushed_prefix_messages", 0),
            flushed_prefix_messages_present=("flushed_prefix_messages" in d),
            last_request_time=d.get("last_request_time", 0.0),
            last_compacted_turn=d.get("last_compacted_turn", -1),
            last_completed_turn=d.get("last_completed_turn", -1),
            last_indexed_turn=d.get("last_indexed_turn", -1),
            checkpoint_version=d.get("checkpoint_version", 0),
            conversation_generation=d.get("conversation_generation", 0),
            tool_tag_counter=d.get("tool_tag_counter", 0),
            split_processed_tags=set(d.get("split_processed_tags", [])),
            trailing_fingerprint=d.get("trailing_fingerprint", ""),
            provider=d.get("provider", ""),
            session_state=d.get("session_state", ""),
            live_turn_count=d.get("live_turn_count", 0),
            history_message_count=d.get("history_message_count", 0),
            ingestion_done=d.get("ingestion_done", 0),
            ingestion_total=d.get("ingestion_total", 0),
            last_payload_kb=d.get("last_payload_kb", 0.0),
            last_payload_tokens=d.get("last_payload_tokens", 0),
            raw_payload_entry_count=d.get("raw_payload_entry_count", 0),
            ingestible_entry_count=d.get("ingestible_entry_count", 0),
            skipped_payload_entry_count=d.get("skipped_payload_entry_count", 0),
            turn_tag_entries=d.get("turn_tag_entries", []),
            working_set=d.get("working_set", []),
            telemetry_rollup=d.get("telemetry_rollup", {}),
            request_captures=d.get("request_captures", []),
            version=d.get("version", 0),
            deleted=d.get("deleted", False),
        )


# The tag-vector runtime cache is PROCESS-GLOBAL: values are keyed by
# (model_name, tag) and are content-derived — the same tag text embeds to
# the same vector for every conversation and tenant — so there is nothing
# per-instance about them, while the cost of materializing them from the
# shared cache is CPU that scales with vocabulary and was being paid per
# provider instance. Tag NAMES are user content; holding them process-wide
# is the same exposure class as the shared cache one layer down. Mutation
# happens under the lock; readers receive copies.
_PROCESS_TAG_VECTOR_CACHE: dict[str, OrderedDict[str, list[float]]] = {}
_PROCESS_TAG_VECTOR_LOCK = threading.Lock()


class SessionStateProvider:
    """Redis-backed session state. Load at request start, save at request end."""

    _PAYLOAD_TOKEN_CACHE_TTL_SECONDS = 6 * 60 * 60
    _TAG_STATS_CACHE_TTL_SECONDS = 6 * 60 * 60
    _TAG_EMBEDDING_CACHE_TTL_SECONDS = 24 * 60 * 60
    _TAG_SUMMARY_EMBEDDING_SNAPSHOT_TTL_SECONDS = 24 * 60 * 60
    _CONTEXT_HINT_CACHE_TTL_SECONDS = 6 * 60 * 60
    # Default sized against measured RUNTIME vocabularies, which run about
    # a thousand entries above durable counts (live, uncompacted tags):
    # the largest observed conversation holds ~7,000 runtime entries, so
    # 10,000 keeps every measured single conversation resident with margin
    # while bounding worst-case per-process residency. Several outlier
    # conversations sharing one process can exceed it and degrade to LRU
    # re-materialization; raise via the environment override if that shows
    # up in the embedding breakdown instrumentation.
    _TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL = 10000
    _TAG_EMBEDDING_RUNTIME_MAX_ENV = "VC_TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL"

    def __init__(
        self,
        redis_client=None,
        redis_url: str = "",
        store=None,
        tag_embedding_runtime_max_per_model: int | None = None,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        else:
            raise ValueError("redis_client or redis_url required")
        self._store = store  # Optional ContextStore for Postgres backup/fallback
        self._degraded = False
        # The per-model runtime cache bound must hold the live tag
        # vocabulary, or every request pays a re-materialization of the
        # entries the previous request evicted. Resolved ONCE at
        # construction (argument, then environment, then the class
        # default) so a mid-flight environment change cannot alter
        # behavior; an invalid value fails loudly rather than silently
        # running with a wrong bound.
        resolved_cap = tag_embedding_runtime_max_per_model
        if resolved_cap is None:
            import os as _os
            raw_cap = _os.environ.get(
                self._TAG_EMBEDDING_RUNTIME_MAX_ENV, "",
            ).strip()
            if raw_cap:
                resolved_cap = int(raw_cap)  # ValueError on garbage
        if resolved_cap is None:
            resolved_cap = self._TAG_EMBEDDING_RUNTIME_MAX_PER_MODEL
        if int(resolved_cap) < 1:
            raise ValueError(
                "tag_embedding_runtime_max_per_model must be a positive "
                f"integer, got {resolved_cap!r}"
            )
        self._tag_embedding_runtime_max_per_model = int(resolved_cap)
        self._tag_stats_runtime_cache: dict[str, list[TagStats]] = {}
        self._tag_summary_embedding_snapshot_runtime_cache: dict[str, dict[str, list[float]]] = {}

    def _key(self, conversation_id: str) -> str:
        return f"vc:session:{conversation_id}"

    def _payload_token_cache_key(self, conversation_id: str, scope: str = "inbound") -> str:
        return f"vc:payload_tokens:{scope}:{conversation_id}"

    def _tag_stats_cache_key(self, conversation_id: str) -> str:
        return f"vc:tag_stats:{conversation_id}"

    def _tag_embedding_cache_key(self, model_name: str, tag: str) -> str:
        digest = hashlib.sha1(f"{model_name}\0{tag}".encode("utf-8")).hexdigest()
        return f"vc:tag_embedding:{digest}"

    def _tag_summary_embedding_snapshot_key(self, conversation_id: str) -> str:
        return f"vc:tag_summary_embeddings:{conversation_id}"

    def _context_hint_cache_key(self, conversation_id: str, cache_key: str) -> str:
        return f"vc:context_hint:{conversation_id}:{cache_key}"

    @property
    def _tag_embedding_runtime_cache(
        self,
    ) -> dict[str, OrderedDict[str, list[float]]]:
        """Alias for the process-global vector cache.

        Kept as an instance-shaped surface so existing callers (and the
        clearing pattern in tests) keep working; there is exactly one
        store per process.
        """
        return _PROCESS_TAG_VECTOR_CACHE

    def _runtime_tag_cache(self, model_name: str) -> OrderedDict[str, list[float]]:
        cache = _PROCESS_TAG_VECTOR_CACHE.get(model_name)
        if cache is None:
            with _PROCESS_TAG_VECTOR_LOCK:
                cache = _PROCESS_TAG_VECTOR_CACHE.get(model_name)
                if cache is None:
                    cache = OrderedDict()
                    _PROCESS_TAG_VECTOR_CACHE[model_name] = cache
        return cache

    def _remember_runtime_tag_embedding(
        self,
        model_name: str,
        tag: str,
        embedding: list[float],
    ) -> None:
        # The store is shared across every provider and engine in the
        # process, so all mutation happens under the lock. The bound is the
        # inserting provider's resolved bound; deployments configure one
        # value per process.
        cache = self._runtime_tag_cache(model_name)
        with _PROCESS_TAG_VECTOR_LOCK:
            cache[tag] = list(embedding)
            cache.move_to_end(tag)
            while len(cache) > self._tag_embedding_runtime_max_per_model:
                cache.popitem(last=False)

    @staticmethod
    def _clone_tag_stats(stats: list[TagStats]) -> list[TagStats]:
        return [
            TagStats(
                tag=item.tag,
                usage_count=item.usage_count,
                total_full_tokens=item.total_full_tokens,
                total_summary_tokens=item.total_summary_tokens,
                oldest_segment=item.oldest_segment,
                newest_segment=item.newest_segment,
            )
            for item in stats
        ]

    @staticmethod
    def _clone_embedding_map(
        embeddings: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        return {tag: list(values) for tag, values in embeddings.items()}

    @staticmethod
    def _parse_datetime(value):
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    @staticmethod
    def _serialize_tag_stats(stats: list[TagStats]) -> list[dict]:
        return [
            {
                "tag": item.tag,
                "usage_count": item.usage_count,
                "total_full_tokens": item.total_full_tokens,
                "total_summary_tokens": item.total_summary_tokens,
                "oldest_segment": item.oldest_segment.isoformat() if item.oldest_segment else None,
                "newest_segment": item.newest_segment.isoformat() if item.newest_segment else None,
            }
            for item in stats
        ]

    @classmethod
    def _deserialize_tag_stats(cls, payload: list[dict] | None) -> list[TagStats]:
        stats: list[TagStats] = []
        for row in payload or []:
            if not isinstance(row, dict):
                continue
            stats.append(TagStats(
                tag=str(row.get("tag", "")),
                usage_count=int(row.get("usage_count", 0) or 0),
                total_full_tokens=int(row.get("total_full_tokens", 0) or 0),
                total_summary_tokens=int(row.get("total_summary_tokens", 0) or 0),
                oldest_segment=cls._parse_datetime(row.get("oldest_segment")),
                newest_segment=cls._parse_datetime(row.get("newest_segment")),
            ))
        return stats

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        if not embedding:
            return []
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if norm == 0.0:
            return [float(value) for value in embedding]
        return [float(value) / norm for value in embedding]

    def get_marker(self, conversation_id: str, marker_name: str):
        """Fast Redis read of a single SessionState field.

        Used by the cross-channel-mirror Tier 2 staleness check, which
        needs a single marker (typically ``last_completed_turn``) on
        every participant-conv inbound request. Performing the full
        ``load`` deserialization round-trip there is wasteful because
        the full SessionState dataclass carries `turn_tag_entries`,
        `telemetry_rollup`, `request_captures`, and timestamp parsing
        that the gate does not consult. ``get_marker`` does one Redis
        ``GET`` and a lightweight JSON key extract.

        Returns:
            The raw decoded marker value (typically ``int`` for the
            turn-number markers), or ``None`` when the SessionState is
            absent / Redis is degraded / the blob is malformed / the
            named field is missing. Callers MUST tolerate ``None`` and
            fall through to a safe default (the Tier 2 caller falls
            through to Tier 3 unconditionally on ``None``).
        """
        if not conversation_id or not marker_name:
            return None
        try:
            raw = self._redis.get(self._key(conversation_id))
        except Exception:
            logger.warning(
                "Redis get_marker failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            self._degraded = True
            return None
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data.get(marker_name)

    def load(self, conversation_id: str) -> SessionState | None:
        """Load session state from Redis. Returns None if not found.
        Returns SessionState(deleted=True) if tombstoned.
        """
        try:
            raw = self._redis.get(self._key(conversation_id))
            # A successful GET -- including a clean miss -- proves Redis
            # recovered after a prior transport failure. ``_degraded`` tracks
            # Redis authority, not whether the optional durable fallback has
            # a row.
            self._degraded = False
            if raw is None:
                # Redis miss — try Postgres fallback
                return self._load_from_store(conversation_id)
            return SessionState.from_json(raw)
        except Exception:
            logger.warning("Redis load failed for %s", conversation_id[:12], exc_info=True)
            self._degraded = True
            # Degraded — try Postgres fallback
            return self._load_from_store(conversation_id)

    def load_authoritative(self, conversation_id: str) -> SessionState | None:
        """Load only from Redis, never from the durable fallback.

        This per-call result is safe for multiworker lifecycle decisions.
        Callers must not infer whether *their* read used Redis by sampling the
        provider-wide ``is_degraded`` flag after ``load()``: another request
        can clear that shared flag while the failed call is still completing
        its fallback.
        """
        _raw, state = self.load_authoritative_snapshot(conversation_id)
        return state

    def load_authoritative_snapshot(
        self,
        conversation_id: str,
    ) -> tuple[bytes | None, SessionState | None]:
        """Return the exact Redis preimage and its decoded state.

        This is intentionally separate from :meth:`load`: administrative
        repair code must never mistake a durable fallback for the Redis value
        it is about to compare-and-swap.
        """
        try:
            raw = self._redis.get(self._key(conversation_id))
            if raw is None:
                self._degraded = False
                return None, None
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            state = SessionState.from_json(raw)
            self._degraded = False
            return raw, state
        except Exception:
            self._degraded = True
            logger.warning(
                "Authoritative Redis load failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            raise

    def repair_session_state_markers(
        self,
        conversation_id: str,
        *,
        expected_raw: bytes | None,
        markers: SessionState,
        durable_generation: int,
        allow_generation_promotion: bool = False,
    ) -> int:
        """Repair derivable markers using an exact-preimage lifecycle CAS.

        Ordinary :meth:`save` remains generation-strict. A populated older-
        generation checkpoint may advance only when the explicit admin caller
        sets ``allow_generation_promotion=True``; normal VCATTACH repair is
        equal-generation only. The
        method acquires the same Redis lifecycle lease used by cloud
        delete/recreate operations, requires the exact authoritative Redis
        preimage, rejects tombstones and generation downgrades, preserves all
        unrelated and unknown JSON fields, and changes only database-derived
        markers plus checkpoint/version counters and generation.
        """
        import redis
        import uuid

        if type(durable_generation) is not int or durable_generation < 0:
            raise RuntimeError("durable conversation generation is invalid")
        if type(allow_generation_promotion) is not bool:
            raise RuntimeError("generation-promotion policy is invalid")
        if (
            type(markers.conversation_generation) is not int
            or markers.conversation_generation != durable_generation
        ):
            raise RuntimeError("marker state generation is not durable-current")

        store = self._store
        get_generation = getattr(store, "get_conversation_generation", None)
        is_deleted = getattr(store, "is_conversation_deleted", None)
        if not callable(get_generation) or not callable(is_deleted):
            raise RuntimeError("durable conversation lifecycle API is unavailable")

        def require_active_generation() -> None:
            deleted = is_deleted(conversation_id)
            if deleted is True:
                raise RuntimeError("refusing marker repair for deleted conversation")
            if deleted is not False:
                raise RuntimeError("durable conversation lifecycle state is invalid")
            observed = get_generation(conversation_id)
            if type(observed) is not int or observed < 0:
                raise RuntimeError("durable conversation generation is invalid")
            if observed != durable_generation:
                raise RuntimeError(
                    "durable conversation generation changed during marker repair"
                )

        expected = expected_raw
        if isinstance(expected, str):
            expected = expected.encode("utf-8")
        key = self._key(conversation_id)
        lease_key = f"vc:lifecycle_lease:{conversation_id}"

        def attempt_commit_under_lease(token: str) -> int:
            committed: SessionState | None = None
            with self._redis.pipeline() as pipe:
                pipe.watch(lease_key, key)
                owner = pipe.get(lease_key)
                if isinstance(owner, bytes):
                    owner = owner.decode("utf-8")
                if owner != token:
                    pipe.unwatch()
                    raise RuntimeError("conversation lifecycle lease was lost")

                current_raw = pipe.get(key)
                if isinstance(current_raw, str):
                    current_raw = current_raw.encode("utf-8")
                if current_raw != expected:
                    pipe.unwatch()
                    raise RuntimeError(
                        "authoritative Redis state changed before marker repair"
                    )

                if current_raw is None:
                    payload = json.loads(markers.to_json())
                    current_version = 0
                    current_checkpoint = 0
                else:
                    payload = json.loads(current_raw)
                    if not isinstance(payload, dict):
                        pipe.unwatch()
                        raise RuntimeError("authoritative Redis state is not an object")
                    deleted = payload.get("deleted", False)
                    if type(deleted) is not bool:
                        pipe.unwatch()
                        raise RuntimeError(
                            "authoritative Redis deletion marker is invalid"
                        )
                    if deleted:
                        pipe.unwatch()
                        raise RuntimeError(
                            "refusing to replace authoritative Redis tombstone"
                        )
                    redis_generation = payload.get("conversation_generation", 0)
                    if type(redis_generation) is not int or redis_generation < 0:
                        pipe.unwatch()
                        raise RuntimeError(
                            "authoritative Redis generation is invalid"
                        )
                    if redis_generation > durable_generation:
                        pipe.unwatch()
                        raise RuntimeError(
                            "refusing session-state generation downgrade"
                        )
                    if (
                        redis_generation < durable_generation
                        and not allow_generation_promotion
                    ):
                        pipe.unwatch()
                        raise RuntimeError(
                            "refusing non-administrative session-state "
                            "generation promotion"
                        )
                    current_version = payload.get("version", 0)
                    current_checkpoint = payload.get("checkpoint_version", 0)
                if type(current_version) is not int or current_version < 0:
                    pipe.unwatch()
                    raise RuntimeError("authoritative Redis version is invalid")
                if (
                    type(current_checkpoint) is not int
                    or current_checkpoint < 0
                ):
                    pipe.unwatch()
                    raise RuntimeError(
                        "authoritative Redis checkpoint version is invalid"
                    )
                payload.update({
                    "compacted_prefix_messages": int(
                        markers.compacted_prefix_messages
                    ),
                    "flushed_prefix_messages": int(
                        markers.flushed_prefix_messages
                    ),
                    "last_compacted_turn": int(markers.last_compacted_turn),
                    "last_completed_turn": int(markers.last_completed_turn),
                    "last_indexed_turn": int(markers.last_indexed_turn),
                    "turn_tag_entries": list(markers.turn_tag_entries),
                    "conversation_generation": durable_generation,
                    "checkpoint_version": current_checkpoint + 1,
                    "version": current_version + 1,
                    "deleted": False,
                })
                encoded = json.dumps(
                    payload,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
                require_active_generation()
                pipe.multi()
                pipe.set(key, encoded)
                pipe.execute()
                committed = SessionState.from_json(encoded)

            require_active_generation()
            self._degraded = False
            assert committed is not None
            self._save_to_store(conversation_id, committed)
            return int(committed.version)

        def commit_under_lease(token: str) -> int:
            for _attempt in range(3):
                try:
                    return attempt_commit_under_lease(token)
                except redis.WatchError:
                    # Cloud renews the watched lifecycle key periodically.
                    # Retry only while the next attempt still proves the same
                    # lease owner and exact session preimage.
                    continue
            raise RuntimeError(
                "session-state marker repair lost repeated Redis races"
            )

        # Cloud's provider exposes a renewable, thread-reentrant lifecycle
        # lease. Use it when available so VCATTACH can safely call this method
        # while already holding the same conversation lease. The standalone
        # core provider has no such helper, so administrative CLI repair takes
        # the exact same Redis lease key directly for this bounded CAS.
        lifecycle_lease = getattr(self, "lifecycle_lease", None)
        if callable(lifecycle_lease):
            with lifecycle_lease(conversation_id) as token:
                require_active_generation()
                return commit_under_lease(token)

        require_active_generation()
        token = uuid.uuid4().hex
        acquired = self._redis.set(lease_key, token, nx=True, px=30_000)
        if not acquired:
            raise RuntimeError("conversation lifecycle lease is busy")
        try:
            return commit_under_lease(token)
        finally:
            try:
                with self._redis.pipeline() as pipe:
                    pipe.watch(lease_key)
                    owner = pipe.get(lease_key)
                    if isinstance(owner, bytes):
                        owner = owner.decode("utf-8")
                    if owner == token:
                        pipe.multi()
                        pipe.delete(lease_key)
                        pipe.execute()
                    else:
                        pipe.unwatch()
            except Exception:
                logger.warning(
                    "Failed to release marker-repair lease for %s",
                    conversation_id[:12],
                    exc_info=True,
                )

    def save(self, conversation_id: str, state: SessionState) -> int | None:
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
        original_version = int(getattr(state, "version", 0) or 0)
        next_version = original_version + 1
        try:
            with self._redis.pipeline() as pipe:
                pipe.watch(key)
                current_raw = pipe.get(key)
                if current_raw:
                    current = json.loads(current_raw)
                    if current.get("deleted"):
                        state.version = original_version
                        logger.info("Save rejected for %s — tombstoned", conversation_id[:12])
                        return None
                    current_generation = int(
                        current.get("conversation_generation", 0) or 0
                    )
                    state_generation = int(
                        getattr(state, "conversation_generation", 0) or 0
                    )
                    if current_generation != state_generation:
                        state.version = original_version
                        logger.info(
                            "Save rejected for %s — stale generation %d != %d",
                            conversation_id[:12],
                            state_generation,
                            current_generation,
                        )
                        return None
                    current_version = int(current.get("version", 0) or 0)
                    if current_version > original_version:
                        state.version = original_version
                        logger.info(
                            "Save rejected for %s — stale version %d < %d",
                            conversation_id[:12],
                            original_version,
                            current_version,
                        )
                        return None
                state.version = next_version
                pipe.multi()
                pipe.set(key, state.to_json())
                pipe.execute()
            self._degraded = False
            # Postgres backup only when Redis succeeded — if Redis failed,
            # writing to Postgres would put it ahead of Redis, and load()
            # would later trust the stale Redis copy over the newer store.
            self._save_to_store(conversation_id, state)
            return next_version
        except Exception:
            state.version = original_version
            logger.warning("Redis save failed for %s — skipping Postgres backup",
                           conversation_id[:12], exc_info=True)
            self._degraded = True
            return None

    def load_payload_token_cache(self, conversation_id: str, *, scope: str = "inbound"):
        """Load the segmented inbound token cache for a conversation.

        This cache is an optional hot-path optimization only. Failures should
        never affect correctness or the primary session-state flow.
        """
        try:
            raw = self._redis.get(self._payload_token_cache_key(conversation_id, scope))
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

    def load_tag_stats_snapshot(self, conversation_id: str) -> list[TagStats] | None:
        """Load cached conversation-scoped TagStats snapshot."""
        if conversation_id in self._tag_stats_runtime_cache:
            return self._clone_tag_stats(self._tag_stats_runtime_cache[conversation_id])
        try:
            raw = self._redis.get(self._tag_stats_cache_key(conversation_id))
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            stats = self._deserialize_tag_stats(json.loads(raw))
            self._tag_stats_runtime_cache[conversation_id] = self._clone_tag_stats(stats)
            return stats
        except Exception:
            logger.warning(
                "Redis tag-stats cache load failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            return None

    def save_tag_stats_snapshot(
        self,
        conversation_id: str,
        tag_stats: list[TagStats],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Save shared TagStats snapshot for a conversation."""
        try:
            serialized = self._serialize_tag_stats(tag_stats)
            self._tag_stats_runtime_cache[conversation_id] = self._deserialize_tag_stats(serialized)
            self._redis.set(
                self._tag_stats_cache_key(conversation_id),
                json.dumps(serialized, default=str).encode("utf-8"),
                ex=ttl_seconds or self._TAG_STATS_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Redis tag-stats cache save failed for %s",
                conversation_id[:12],
                exc_info=True,
            )

    def refresh_tag_stats_snapshot(self, conversation_id: str) -> list[TagStats] | None:
        """Rebuild the shared TagStats snapshot from the backing store."""
        if self._store is None or not hasattr(self._store, "get_all_tags"):
            return None
        tag_stats = self._store.get_all_tags(conversation_id=conversation_id)
        self.save_tag_stats_snapshot(conversation_id, tag_stats)
        return self._clone_tag_stats(tag_stats)

    def delete_tag_stats_snapshot(self, conversation_id: str) -> None:
        self._tag_stats_runtime_cache.pop(conversation_id, None)
        try:
            self._redis.delete(self._tag_stats_cache_key(conversation_id))
        except Exception:
            pass

    def save_payload_token_cache(
        self,
        conversation_id: str,
        cache,
        *,
        scope: str = "inbound",
        ttl_seconds: int | None = None,
    ) -> None:
        """Save the segmented inbound token cache for a conversation.

        Stored separately from durable session state so it can be updated on
        every request without inflating the authoritative checkpoint blob.
        """
        if cache is None:
            return
        try:
            payload = asdict(cache) if hasattr(cache, "__dataclass_fields__") else cache
            self._redis.set(
                self._payload_token_cache_key(conversation_id, scope),
                json.dumps(payload, default=str).encode("utf-8"),
                ex=ttl_seconds or self._PAYLOAD_TOKEN_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Redis payload-token cache save failed for %s",
                conversation_id[:12],
                exc_info=True,
            )

    def delete_payload_token_cache(self, conversation_id: str, *, scope: str = "inbound") -> None:
        """Best-effort delete for the segmented inbound token cache."""
        try:
            self._redis.delete(self._payload_token_cache_key(conversation_id, scope))
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

        loaded: dict[str, list[float]] = {}
        runtime_cache = self._runtime_tag_cache(model_name)
        missing: list[str] = []
        with _PROCESS_TAG_VECTOR_LOCK:
            for tag in unique_tags:
                cached = runtime_cache.get(tag)
                if cached is None:
                    missing.append(tag)
                    continue
                runtime_cache.move_to_end(tag)
                loaded[tag] = list(cached)
        if not missing:
            return loaded

        try:
            keys = [self._tag_embedding_cache_key(model_name, tag) for tag in missing]
            mget = getattr(self._redis, "mget", None)
            if callable(mget):
                raw_values = mget(keys)
            else:
                raw_values = [self._redis.get(key) for key in keys]

            for tag, raw in zip(missing, raw_values):
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                value = json.loads(raw)
                if isinstance(value, list):
                    loaded[tag] = value
                    self._remember_runtime_tag_embedding(model_name, tag, value)
            return loaded
        except Exception:
            logger.warning(
                "Redis tag-embedding cache load failed (model=%s tags=%d)",
                model_name,
                len(missing),
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
                    self._remember_runtime_tag_embedding(model_name, tag, embedding)
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

    def load_tag_summary_embedding_snapshot(
        self,
        conversation_id: str,
    ) -> dict[str, list[float]] | None:
        """Load cached normalized tag-summary embeddings for a conversation."""
        if conversation_id in self._tag_summary_embedding_snapshot_runtime_cache:
            return self._clone_embedding_map(
                self._tag_summary_embedding_snapshot_runtime_cache[conversation_id]
            )
        try:
            raw = self._redis.get(self._tag_summary_embedding_snapshot_key(conversation_id))
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return None
            normalized = {
                str(tag): self._normalize_embedding(list(values))
                for tag, values in parsed.items()
                if isinstance(values, list)
            }
            self._tag_summary_embedding_snapshot_runtime_cache[conversation_id] = (
                self._clone_embedding_map(normalized)
            )
            return normalized
        except Exception:
            logger.warning(
                "Redis tag-summary embedding snapshot load failed for %s",
                conversation_id[:12],
                exc_info=True,
            )
            return None

    def save_tag_summary_embedding_snapshot(
        self,
        conversation_id: str,
        embeddings_by_tag: dict[str, list[float]],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Save normalized tag-summary embeddings for retrieval scoring."""
        try:
            normalized = {
                str(tag): self._normalize_embedding(list(values))
                for tag, values in embeddings_by_tag.items()
                if isinstance(values, list)
            }
            self._tag_summary_embedding_snapshot_runtime_cache[conversation_id] = (
                self._clone_embedding_map(normalized)
            )
            self._redis.set(
                self._tag_summary_embedding_snapshot_key(conversation_id),
                json.dumps(normalized, default=str).encode("utf-8"),
                ex=ttl_seconds or self._TAG_SUMMARY_EMBEDDING_SNAPSHOT_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Redis tag-summary embedding snapshot save failed for %s",
                conversation_id[:12],
                exc_info=True,
            )

    def refresh_tag_summary_embedding_snapshot(
        self,
        conversation_id: str,
    ) -> dict[str, list[float]] | None:
        """Rebuild the shared tag-summary embedding snapshot from the store."""
        if self._store is None or not hasattr(self._store, "load_tag_summary_embeddings"):
            return None
        embeddings = self._store.load_tag_summary_embeddings(conversation_id=conversation_id)
        self.save_tag_summary_embedding_snapshot(conversation_id, embeddings)
        return self._clone_embedding_map(
            self._tag_summary_embedding_snapshot_runtime_cache.get(conversation_id, {})
        )

    def delete_tag_summary_embedding_snapshot(self, conversation_id: str) -> None:
        self._tag_summary_embedding_snapshot_runtime_cache.pop(conversation_id, None)
        try:
            self._redis.delete(self._tag_summary_embedding_snapshot_key(conversation_id))
        except Exception:
            pass

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

    def publish_tombstone(self, conversation_id: str) -> None:
        """Publish the Redis deletion fence without touching PostgreSQL.

        Cloud lifecycle code uses this before durable purge so other workers
        stop serving the generation before destructive work begins. ``delete``
        calls it too, preserving the original public all-backends contract.
        """
        tombstone = SessionState(deleted=True, version=_MAX_VERSION)
        try:
            self._redis.set(
                self._key(conversation_id),
                tombstone.to_json(),
                ex=86400,
            )
            self.delete_payload_token_cache(conversation_id, scope="inbound")
            self.delete_payload_token_cache(conversation_id, scope="outbound")
            self.delete_tag_stats_snapshot(conversation_id)
            self.delete_tag_summary_embedding_snapshot(conversation_id)
            self._degraded = False
        except Exception as e:
            self._degraded = True
            raise RuntimeError(
                f"Redis tombstone failed for {conversation_id[:12]}: {e}",
            ) from e

    def delete(self, conversation_id: str) -> None:
        """Tombstone the conversation. BOTH backends must succeed.

        Redis tombstone prevents workers from loading the live state.
        Postgres delete prevents resurrection via _load_from_store fallback.
        If either fails, retry it — a partial delete leaks state.
        """
        errors: list[str] = []

        # 1. Redis tombstone
        try:
            self.publish_tombstone(conversation_id)
        except Exception as e:
            errors.append(f"Redis: {e}")

        # 2. Durable delete. Carry the generation returned by the deletion
        # fence into the purge transaction so a stale worker cannot erase a
        # conversation recreated after this delete began.
        if self._store and hasattr(self._store, "delete_conversation"):
            try:
                begin_delete = getattr(
                    self._store,
                    "begin_conversation_deletion",
                    None,
                )
                expected_generation = (
                    int(begin_delete(conversation_id))
                    if callable(begin_delete)
                    else None
                )
                kwargs = (
                    {"expected_generation": expected_generation}
                    if expected_generation is not None
                    else {}
                )
                self._store.delete_conversation(conversation_id, **kwargs)
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

    def undelete(self, conversation_id: str) -> None:
        """Clear a Redis tombstone so the same conversation id can be reused.

        DELETE is still authoritative for existing persisted data. This only
        removes the session-state tombstone fence so a subsequent fresh client
        request can recreate the conversation under the same id.
        """
        self._tag_stats_runtime_cache.pop(conversation_id, None)
        self._tag_summary_embedding_snapshot_runtime_cache.pop(conversation_id, None)
        try:
            activate = getattr(self._store, "activate_conversation", None)
            generation = 0
            if callable(activate):
                generation = int(
                    activate(conversation_id, recreate_deleted=True) or 0
                )
            raw = self._redis.get(self._key(conversation_id))
            current = SessionState.from_json(raw) if raw is not None else None
            if current is None or current.deleted:
                # Keep an active-generation marker instead of deleting the
                # tombstone key.  A detached pre-delete worker then sees a
                # generation mismatch and cannot republish its old checkpoint.
                self._redis.set(
                    self._key(conversation_id),
                    SessionState(
                        conversation_generation=generation,
                        version=0,
                    ).to_json(),
                )
            self._degraded = False
        except Exception:
            logger.warning("Redis undelete failed for %s", conversation_id[:12], exc_info=True)
            self._degraded = True

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
        is_deleted = getattr(self._store, "is_conversation_deleted", None)
        if callable(is_deleted):
            try:
                if is_deleted(conversation_id) is True:
                    logger.info(
                        "Skipping durable fallback for deleted conversation %s",
                        conversation_id[:12],
                    )
                    return SessionState(deleted=True, version=_MAX_VERSION)
            except Exception:
                logger.warning(
                    "Skipping durable fallback for %s — lifecycle state unavailable",
                    conversation_id[:12],
                    exc_info=True,
                )
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
                canonical_turn_id=d.get("canonical_turn_id", "") or "",
                tags=d.get("tags", []),
                primary_tag=d.get("primary_tag", ""),
                message_hash=d.get("message_hash", ""),
                sender=d.get("sender", ""),
                timestamp=ts,
                session_date=d.get("session_date", ""),
                fact_signals=fs,
                code_refs=d.get("code_refs", []) or [],
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
            compacted_prefix_messages=state.compacted_prefix_messages,
            flushed_prefix_messages=state.flushed_prefix_messages,
            flushed_prefix_messages_present=state.flushed_prefix_messages_present,
            last_request_time=state.last_request_time,
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
                "canonical_turn_id": getattr(e, "canonical_turn_id", "") or "",
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
                "code_refs": list(getattr(e, "code_refs", []) or []),
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
            compacted_prefix_messages=snapshot.compacted_prefix_messages,
            flushed_prefix_messages=getattr(snapshot, 'flushed_prefix_messages', 0),
            flushed_prefix_messages_present=getattr(snapshot, 'flushed_prefix_messages_present', True),
            last_request_time=getattr(snapshot, 'last_request_time', 0.0),
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
