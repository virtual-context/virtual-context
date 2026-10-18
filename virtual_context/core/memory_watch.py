"""Per-operation process memory, logged as ``MEM_OP`` lines.

Each wrapped operation logs resident memory before and after it and the
process peak while it ran. The peak comes from the kernel's resident
high-water mark (``VmHWM``), reset when an operation starts while no other
is running, so with no overlap it is that operation's own peak. Operations
that ran at the same time on other threads are named in ``overlap``; the peak
then covers all of them. An operation nested inside another on the same
thread reports the peak since the outer one started. Only Linux exposes
these counters; elsewhere nothing is logged.

Lines are written when the operation's peak or retained growth reaches
``VC_MEM_OP_LOG_MIN_MB`` (default 10).
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import threading
from contextlib import contextmanager
from typing import Callable

logger = logging.getLogger(__name__)

_STATUS = "/proc/self/status"
_CLEAR_REFS = "/proc/self/clear_refs"
_lock = threading.Lock()
_active: dict[int, tuple[str, int]] = {}  # token -> (op, thread id)
_ids = itertools.count()
_overlaps: dict[int, set[str]] = {}


def _status_mb() -> tuple[float, float] | None:
    """``(VmRSS, VmHWM)`` in MB, or None where the counters are unavailable."""
    try:
        with open(_STATUS) as handle:
            values = {}
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    values[key] = int(rest.split()[0]) / 1024
            return values["VmRSS"], values["VmHWM"]
    except (OSError, KeyError, ValueError):
        return None


def _reset_peak() -> bool:
    try:
        with open(_CLEAR_REFS, "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def _min_mb() -> float:
    try:
        return float(os.environ.get("VC_MEM_OP_LOG_MIN_MB", "10"))
    except ValueError:
        return 10.0


@contextmanager
def memory_operation(op: str, conversation_id: str = ""):
    """Log memory for the enclosed operation (see module docstring)."""
    if _status_mb() is None:
        yield
        return
    token = next(_ids)
    thread = threading.get_ident()
    with _lock:
        alone = not _active
        exact_peak = _reset_peak() if alone else False
        for other, (_name, other_thread) in _active.items():
            if other_thread != thread:
                _overlaps.setdefault(other, set()).add(op)
        _overlaps[token] = {name for name, other_thread in _active.values() if other_thread != thread}
        _active[token] = (op, thread)
        before = _status_mb()
    try:
        yield
    finally:
        with _lock:
            after = _status_mb()
            _active.pop(token, None)
            overlap = sorted(_overlaps.pop(token, set()))
        if before is None or after is None:
            return
        rss_before, _ = before
        rss_after, peak = after
        grew = peak - rss_before
        kept = rss_after - rss_before
        if max(grew, kept) < _min_mb():
            return
        logger.info(
            "MEM_OP op=%s conv=%s rss_before_mb=%.0f peak_mb=%.0f rss_after_mb=%.0f "
            "peak_growth_mb=%.0f kept_mb=%.0f peak_exact=%s overlap=%s",
            op, (conversation_id or "")[:12], rss_before, peak, rss_after,
            grew, kept, "yes" if exact_peak and not overlap else "no",
            ",".join(overlap) or "-",
        )


def measured(op: str | Callable[..., str], conversation_id: Callable[..., str] | None = None):
    """Decorator form of :func:`memory_operation`.

    *op* is a label or a callable receiving the call's arguments;
    *conversation_id* likewise receives them and returns the id to log.
    """
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                label = op(*args, **kwargs) if callable(op) else op
                conv = conversation_id(*args, **kwargs) if conversation_id else ""
            except Exception:
                label, conv = (op if isinstance(op, str) else fn.__name__), ""
            with memory_operation(label, conv):
                return fn(*args, **kwargs)
        return inner
    return wrap
