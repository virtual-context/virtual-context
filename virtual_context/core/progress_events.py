from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    conversation_id: str
    lifecycle_epoch: int
    kind: str  # "ingestion" | "compaction" | "phase_transition" | "lifecycle_reset"
    timestamp: float


@dataclass(frozen=True)
class IngestionProgressEvent(ProgressEvent):
    episode_id: str
    done: int
    total: int


@dataclass(frozen=True)
class CompactionProgressEvent(ProgressEvent):
    operation_id: str
    phase_name: str
    phase_index: int
    phase_count: int
    status: str


@dataclass(frozen=True)
class PhaseTransitionEvent(ProgressEvent):
    old_phase: str
    new_phase: str


@dataclass(frozen=True)
class LifecycleResetEvent(ProgressEvent):
    old_epoch: int
    new_epoch: int
