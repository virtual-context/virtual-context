import threading
from virtual_context.core.event_bus import ProgressEventBus
from virtual_context.core.progress_events import IngestionProgressEvent


def _event(done=1, total=10):
    return IngestionProgressEvent(
        conversation_id="c", lifecycle_epoch=1, kind="ingestion",
        timestamp=1.0, episode_id="e", done=done, total=total,
    )


def test_subscribers_receive_published_events():
    bus = ProgressEventBus()
    received = []
    bus.subscribe(received.append)
    bus.publish(_event())
    assert len(received) == 1
    assert received[0].done == 1


def test_multiple_subscribers_all_receive_events():
    bus = ProgressEventBus()
    a, b = [], []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    bus.publish(_event(done=5))
    assert len(a) == 1 and len(b) == 1
    assert a[0].done == 5 == b[0].done


def test_unsubscribe_stops_delivery():
    bus = ProgressEventBus()
    received = []
    bus.subscribe(received.append)
    bus.unsubscribe(received.append)
    bus.publish(_event())
    assert received == []


def test_unsubscribe_of_unregistered_callback_is_noop():
    bus = ProgressEventBus()
    def cb(_): pass
    # Should not raise.
    bus.unsubscribe(cb)


def test_publish_is_thread_safe():
    bus = ProgressEventBus()
    received = []
    lock = threading.Lock()
    def on_event(e):
        with lock:
            received.append(e)
    bus.subscribe(on_event)
    threads = [
        threading.Thread(target=lambda i=i: bus.publish(_event(done=i)))
        for i in range(100)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(received) == 100


def test_subscriber_exceptions_do_not_break_other_subscribers():
    bus = ProgressEventBus()
    received = []
    def bad(_):
        raise RuntimeError("boom")
    bus.subscribe(bad)
    bus.subscribe(received.append)
    bus.publish(_event())
    assert len(received) == 1


def test_subscriber_modifying_list_during_publish_is_safe():
    """A subscriber that unsubscribes itself during publish must not
    break iteration for the current event delivery."""
    bus = ProgressEventBus()
    received_first = []
    def self_unsub(ev):
        received_first.append(ev)
        bus.unsubscribe(self_unsub)
    bus.subscribe(self_unsub)
    received_second = []
    bus.subscribe(received_second.append)
    bus.publish(_event())
    bus.publish(_event())
    # First publish delivered to both; second publish only to received_second.
    assert len(received_first) == 1
    assert len(received_second) == 2
