import logging
import threading
from typing import Callable

from .progress_events import ProgressEvent

logger = logging.getLogger(__name__)


class ProgressEventBus:
    """In-process, thread-safe publish/subscribe for ProgressEvents.

    Single-process scope. For cross-worker fanout in cloud deployments,
    a Redis bridge subscribes to this bus and republishes to Redis Pub/Sub
    (see Task B2).

    Subscriber exceptions are caught and logged — a broken subscriber
    cannot prevent other subscribers from receiving events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[ProgressEvent], None]] = []

    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def publish(self, event: ProgressEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            try:
                cb(event)
            except Exception:
                logger.exception("ProgressEventBus subscriber raised")
