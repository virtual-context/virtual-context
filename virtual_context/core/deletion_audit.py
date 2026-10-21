"""Audit line for every conversation deletion: who started it and from where."""
from __future__ import annotations

import logging
import os
import threading
import traceback

logger = logging.getLogger("virtual_context.deletion_audit")

_SKIP = ("deletion_audit.py",)


def caller_chain(limit: int = 8) -> str:
    """Return the innermost application frames as ``file:line:function`` links."""
    frames = []
    for frame in reversed(traceback.extract_stack()[:-1]):
        name = os.path.basename(frame.filename)
        if name in _SKIP:
            continue
        if "site-packages" in frame.filename and "virtual_context" not in frame.filename and "vc_cloud" not in frame.filename:
            continue
        frames.append(f"{name}:{frame.lineno}:{frame.name}")
        if len(frames) >= limit:
            break
    return " <- ".join(frames)


def log_deletion(conversation_id: str, generation: int, backend: str) -> None:
    """Log the lifecycle tombstone write with the full id and the call chain."""
    logger.warning(
        "CONVERSATION_DELETE_BEGIN conv=%s generation=%d backend=%s thread=%s pid=%d caller=%s",
        conversation_id, generation, backend, threading.current_thread().name,
        os.getpid(), caller_chain(),
    )
