"""Runtime mode for the compaction operation-id fence.

Per fencing plan §9.0, the fence ships with a tier-rollout discipline
controlled by ``VC_COMPACTION_FENCE_MODE``:

* ``off`` (Tier 0, the deploy default): the new columns + indexes are
  present but the fence machinery is dormant. Per-write guard
  rejections do not raise, the ``facts.operation_id`` stamping
  introduced in commits 14cf5e6 / 5db874f reverts to ``NULL`` on new
  inserts, and the cleanup extension shipped in 4f4ec95 leaves the
  three new tables (``segment_chunks``, ``segment_tool_outputs``,
  ``fact_links``) untouched.
* ``observe`` (Tier 1): writes stamp ``operation_id`` but guard
  rejections only log ``COMPACTION_FENCE_OBSERVED_MISMATCH`` --
  ``CompactionLeaseLost`` is not raised. Cleanup computes
  would-delete counts and logs them without executing the DELETE.
* ``active`` (Tier 2): the full fence as shipped through P1-P6 is
  enforced. Guard rejections raise ``CompactionLeaseLost``, cleanup
  executes the seven-table DELETE, and the C2R gate is respected.

Construction and lifetime:

* The mode is parsed once at store construction by
  ``CompactionFenceMode.from_env()``. Invalid values raise
  ``ValueError`` so a typo cannot silently downgrade a process to a
  weaker mode.
* ``PostgresStore`` and ``SqliteStore`` accept
  ``compaction_fence_mode: CompactionFenceMode | None = None`` on
  their constructors; ``None`` triggers the env parse immediately.
* ``CompositeStore`` accepts the same kwarg; when its delegates
  expose ``_compaction_fence_mode`` they must match the holder
  passed to the composite so cleanup and write paths cannot run on
  different mode sources.
* Mode is static for the lifetime of the store. Rollback by flipping
  the env var requires a process restart or explicit store
  reconstruction; one compaction operation cannot switch enforcement
  mode mid-transaction.

This module is dependency-light on purpose: stores import it during
``__init__`` before any database connection exists, and the rollout
discipline depends on the parser failing fast on bad input.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum


class CompactionFenceMode(str, Enum):
    """Three-tier rollout mode for the compaction operation-id fence.

    Subclasses ``str`` so legacy callers can compare against the raw
    string values (e.g. ``mode == "active"``) without needing to
    import the enum. ``StrEnum`` is Python 3.11+; this base class
    keeps the same surface while staying compatible with the
    project's stated 3.11 baseline.
    """

    OFF = "off"
    OBSERVE = "observe"
    ACTIVE = "active"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "CompactionFenceMode":
        """Parse the mode from ``VC_COMPACTION_FENCE_MODE``.

        The env var is read once at store construction and the
        resolved holder is pinned for the lifetime of the store. A
        missing env var defaults to ``OFF``, the deploy default per
        plan §9.1.

        Accepts case-insensitive values with surrounding whitespace
        for tolerance against shell quoting accidents. Any other
        value raises ``ValueError`` so a typo cannot silently
        downgrade enforcement.
        """
        env = os.environ if environ is None else environ
        raw = env.get("VC_COMPACTION_FENCE_MODE", "off")
        normalized = (raw or "").strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(sorted(member.value for member in cls))
            raise ValueError(
                f"VC_COMPACTION_FENCE_MODE must be one of: {valid}; "
                f"got {raw!r}"
            ) from exc

    @classmethod
    def resolve(
        cls,
        explicit: "CompactionFenceMode | None",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "CompactionFenceMode":
        """Return ``explicit`` when supplied; otherwise parse from
        environment. Centralizes the constructor pattern used by
        ``PostgresStore``, ``SqliteStore``, and ``CompositeStore`` so
        all three resolve the holder identically.
        """
        if explicit is not None:
            return explicit
        return cls.from_env(environ=environ)

    @property
    def is_off(self) -> bool:
        return self is CompactionFenceMode.OFF

    @property
    def is_observe(self) -> bool:
        return self is CompactionFenceMode.OBSERVE

    @property
    def is_active(self) -> bool:
        return self is CompactionFenceMode.ACTIVE

    @property
    def enforces(self) -> bool:
        """True for ACTIVE mode; the per-write fence raises
        ``CompactionLeaseLost`` and cleanup executes the new-table
        DELETEs only at this tier. False for OFF and OBSERVE.
        """
        return self is CompactionFenceMode.ACTIVE

    @property
    def stamps_operation_id(self) -> bool:
        """True for OBSERVE and ACTIVE; the ``facts.operation_id``
        stamp introduced in P3 is written on insert. False for OFF so
        Tier 0 deploys preserve the pre-fence row shape.
        """
        return self is not CompactionFenceMode.OFF


__all__ = ["CompactionFenceMode"]
