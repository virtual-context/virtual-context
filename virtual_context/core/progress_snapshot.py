"""Frozen dataclasses describing a point-in-time progress snapshot.

A ``ProgressSnapshot`` is derived from the ``conversations`` row plus
aggregate SUMs over ``canonical_turns`` and point lookups in
``ingestion_episode`` / ``compaction_operation``.  Consumers treat it as
an immutable view; fields such as ``total_ingestible`` / ``done_ingestible``
are computed at read time so they can never drift from canonical truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveEpisodeSnapshot:
    episode_id: str
    raw_payload_entries: int
    owner_worker_id: str
    heartbeat_ts: str


@dataclass(frozen=True)
class ActiveCompactionSnapshot:
    operation_id: str
    phase_name: str
    phase_index: int
    phase_count: int
    status: str


@dataclass(frozen=True)
class ProgressSnapshot:
    conversation_id: str
    lifecycle_epoch: int
    phase: str
    total_ingestible: int
    done_ingestible: int
    last_raw_payload_entries: int
    last_ingestible_payload_entries: int
    active_episode: ActiveEpisodeSnapshot | None
    active_compaction: ActiveCompactionSnapshot | None
