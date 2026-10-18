"""Per-operation memory lines name each operation's own peak and overlaps."""
from __future__ import annotations

import logging
import threading

import pytest

from virtual_context.core import memory_watch


class _Proc:
    """Fake ``VmRSS``/``VmHWM`` counters."""

    def __init__(self, rss: float):
        self.rss = self.hwm = rss
        self.resets = 0

    def status(self):
        return self.rss, self.hwm

    def reset(self):
        self.resets += 1
        self.hwm = self.rss
        return True

    def grow(self, mb: float):
        self.rss += mb
        self.hwm = max(self.hwm, self.rss)


@pytest.fixture
def proc(monkeypatch):
    fake = _Proc(100)
    monkeypatch.setattr(memory_watch, "_status_mb", fake.status)
    monkeypatch.setattr(memory_watch, "_reset_peak", fake.reset)
    monkeypatch.delenv("VC_MEM_OP_LOG_MIN_MB", raising=False)
    return fake


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEM_OP")]


def test_alone_operation_logs_its_exact_peak(proc, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")
    proc.hwm = 900  # an earlier, unrelated peak
    with memory_watch.memory_operation("compact", "conv-abcdefghijk"):
        proc.grow(300)
        proc.grow(-250)
    (line,) = _lines(caplog)
    assert "op=compact conv=conv-abcdefg " in line
    assert "rss_before_mb=100 peak_mb=400 rss_after_mb=150" in line
    assert "peak_growth_mb=300 kept_mb=50 peak_exact=yes overlap=-" in line


def test_small_operations_are_not_logged(proc, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")
    with memory_watch.memory_operation("inbound"):
        proc.grow(5)
    assert _lines(caplog) == []


def test_operations_on_other_threads_are_named_as_overlap(proc, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")
    started, finish = threading.Event(), threading.Event()

    def other():
        with memory_watch.memory_operation("tool:vc_recall_all"):
            started.set()
            finish.wait(5)
            proc.grow(50)

    thread = threading.Thread(target=other)
    with memory_watch.memory_operation("compact"):
        thread.start()
        started.wait(5)
        proc.grow(100)
        finish.set()
        thread.join(5)
    lines = _lines(caplog)
    compact = next(line for line in lines if "op=compact" in line)
    tool = next(line for line in lines if "op=tool:vc_recall_all" in line)
    assert "peak_exact=no overlap=tool:vc_recall_all" in compact
    assert "peak_exact=no overlap=compact" in tool
    assert proc.resets == 1


def test_nested_operation_on_the_same_thread_is_not_an_overlap(proc, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")
    with memory_watch.memory_operation("inbound"):
        with memory_watch.memory_operation("context_hint"):
            proc.grow(40)
    outer = next(line for line in _lines(caplog) if "op=inbound" in line)
    assert "peak_exact=yes overlap=-" in outer


def test_nothing_is_logged_without_kernel_counters(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")
    monkeypatch.setattr(memory_watch, "_status_mb", lambda: None)
    calls = []

    @memory_watch.measured(lambda name, **k: f"tool:{name}")
    def tool(name, **kwargs):
        calls.append(name)
        return "ok"

    assert tool("vc_find_quote") == "ok" and calls == ["vc_find_quote"]
    assert _lines(caplog) == []


def test_decorator_labels_the_operation(proc, caplog):
    caplog.set_level(logging.INFO, logger="virtual_context.core.memory_watch")

    @memory_watch.measured(lambda engine, name, *a, **k: f"tool:{name}", lambda engine, *a, **k: engine)
    def run(engine, name):
        proc.grow(20)

    run("conv-1", "vc_query_facts")
    (line,) = _lines(caplog)
    assert "op=tool:vc_query_facts conv=conv-1 " in line
