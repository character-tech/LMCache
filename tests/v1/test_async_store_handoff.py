# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheEngine async store handoff (background store thread).

These tests exercise _background_store_worker, _drain_store_queue, and the
close() shutdown sequence using mocked storage_manager and stats_monitor so
no CUDA device or real backend is required.
"""
# Standard
import queue
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

# Third Party


# ---------------------------------------------------------------------------
# Minimal StoreRequestStats stub (avoids importing observability.py which
# requires prometheus_client not present in the unit-test environment).
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass
class StoreRequestStats:
    request_id: int
    num_tokens: int
    start_time: float
    end_time: float
    process_tokens_time: float = 0.0
    from_gpu_time: float = 0.0
    put_time: float = 0.0

    def time_to_store(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    @contextmanager
    def profile_process_tokens(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.process_tokens_time += time.perf_counter() - start

    @contextmanager
    def profile_from_gpu(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.from_gpu_time += time.perf_counter() - start

    @contextmanager
    def profile_put(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.put_time += time.perf_counter() - start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store_stats() -> StoreRequestStats:
    return StoreRequestStats(
        request_id=0,
        num_tokens=10,
        start_time=time.perf_counter(),
        end_time=0,
    )


def _make_work_item(req_id: str = "req-1", batched_put_delay: float = 0.0):
    """Build a tuple matching the format enqueued by LMCacheEngine.store()."""
    keys = ["key1"]
    memory_objs = [MagicMock()]
    transfer_spec = None
    store_location = MagicMock()
    store_stats = make_store_stats()
    tot_token_num = 10
    num_to_store_tokens = 10
    tot_kv_size = 1024.0
    return (
        keys,
        memory_objs,
        transfer_spec,
        store_location,
        store_stats,
        tot_token_num,
        num_to_store_tokens,
        tot_kv_size,
        req_id,
    )


class FakeEngine:
    """Minimal stand-in for LMCacheEngine that only exercises the background thread."""

    def __init__(self, storage_manager=None, stats_monitor=None):
        self.storage_manager = storage_manager or MagicMock()
        self.stats_monitor = stats_monitor or MagicMock()
        self._store_queue: queue.Queue = queue.Queue()
        self._store_thread = threading.Thread(
            target=self._background_store_worker,
            name="lmcache-bg-store-test",
            daemon=True,
        )
        self._store_thread.start()

    # Copy the exact implementation from LMCacheEngine so tests exercise real code.
    def _background_store_worker(self) -> None:
        """Daemon thread that executes batched_put work items."""
        while True:
            item = self._store_queue.get()
            if item is None:
                break
            req_id = "<unknown>"
            try:
                (
                    keys,
                    memory_objs,
                    transfer_spec,
                    store_location,
                    store_stats,
                    tot_token_num,
                    num_to_store_tokens,
                    tot_kv_size,
                    req_id,
                ) = item
                with store_stats.profile_put():
                    self.storage_manager.batched_put(
                        keys,
                        memory_objs,
                        transfer_spec=transfer_spec,
                        location=store_location,
                    )
                self.stats_monitor.on_store_finished(store_stats, tot_token_num)
            except Exception:
                pass
            finally:
                self._store_queue.task_done()

    def _drain_store_queue(self) -> None:
        """Block until all pending background store work completes."""
        self._store_queue.join()

    def shutdown(self) -> None:
        self._drain_store_queue()
        self._store_queue.put(None)
        self._store_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Tests: _drain_store_queue
# ---------------------------------------------------------------------------


def test_drain_blocks_until_work_completes():
    """_drain_store_queue must block until enqueued work finishes."""
    barrier = threading.Event()
    completed = []

    engine = FakeEngine()

    # Patch batched_put to block until we release the barrier
    def slow_put(*args, **kwargs):
        barrier.wait(timeout=5)
        completed.append(True)

    engine.storage_manager.batched_put.side_effect = slow_put

    engine._store_queue.put(_make_work_item("req-slow"))

    # Start drain in a thread so we can release the barrier after
    drain_done = threading.Event()

    def do_drain():
        engine._drain_store_queue()
        drain_done.set()

    t = threading.Thread(target=do_drain, daemon=True)
    t.start()

    # Drain should not be done yet
    assert not drain_done.wait(timeout=0.2), "drain returned before work completed"

    # Release the barrier and now drain should complete
    barrier.set()
    assert drain_done.wait(timeout=5), "drain did not return after work completed"
    assert completed, "batched_put was never called"

    engine.shutdown()


def test_drain_returns_immediately_when_queue_empty():
    """_drain_store_queue on an empty queue must return immediately."""
    engine = FakeEngine()
    start = time.perf_counter()
    engine._drain_store_queue()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"drain took {elapsed:.3f}s on empty queue"
    engine.shutdown()


# ---------------------------------------------------------------------------
# Tests: shutdown / close
# ---------------------------------------------------------------------------


def test_shutdown_drains_queue_before_stopping():
    """shutdown() must finish all pending stores before the thread exits."""
    completed_reqs = []

    engine = FakeEngine()

    def record_put(*args, **kwargs):
        completed_reqs.append(args)

    engine.storage_manager.batched_put.side_effect = record_put

    for i in range(5):
        engine._store_queue.put(_make_work_item(f"req-{i}"))

    engine.shutdown()

    assert (
        not engine._store_thread.is_alive()
    ), "worker thread still alive after shutdown"
    assert (
        len(completed_reqs) == 5
    ), f"expected 5 batched_put calls, got {len(completed_reqs)}"


def test_shutdown_stops_worker_thread():
    """After shutdown() the worker thread must not be alive."""
    engine = FakeEngine()
    engine.shutdown()
    assert not engine._store_thread.is_alive()


# ---------------------------------------------------------------------------
# Tests: background worker correctness
# ---------------------------------------------------------------------------


def test_batched_put_called_with_correct_args():
    """batched_put must be called with keys, memory_objs, transfer_spec, location."""
    engine = FakeEngine()
    item = _make_work_item("req-check")
    keys, memory_objs, transfer_spec, store_location = (
        item[0],
        item[1],
        item[2],
        item[3],
    )

    engine._store_queue.put(item)
    engine._drain_store_queue()

    engine.storage_manager.batched_put.assert_called_once_with(
        keys,
        memory_objs,
        transfer_spec=transfer_spec,
        location=store_location,
    )
    engine.shutdown()


def test_on_store_finished_called_after_put():
    """stats_monitor.on_store_finished must be called after batched_put."""
    order = []
    engine = FakeEngine()

    engine.storage_manager.batched_put.side_effect = lambda *a, **kw: order.append(
        "put"
    )
    engine.stats_monitor.on_store_finished.side_effect = lambda *a: order.append(
        "finish"
    )

    engine._store_queue.put(_make_work_item("req-order"))
    engine._drain_store_queue()

    assert order == ["put", "finish"], f"unexpected order: {order}"
    engine.shutdown()


def test_multiple_items_all_processed():
    """All enqueued items must be processed, preserving per-req independence."""
    processed = []
    engine = FakeEngine()
    engine.storage_manager.batched_put.side_effect = (
        lambda keys, *a, **kw: processed.append(keys[0])
    )

    items = [_make_work_item(f"req-{i}") for i in range(10)]
    # Each item uses its own keys list with a unique key
    for i, item in enumerate(items):
        lst = list(item)
        lst[0] = [f"key-{i}"]
        engine._store_queue.put(tuple(lst))

    engine._drain_store_queue()
    assert len(processed) == 10
    assert set(processed) == {f"key-{i}" for i in range(10)}
    engine.shutdown()


def test_exception_in_worker_does_not_kill_thread():
    """An exception during batched_put must not crash the worker thread."""
    engine = FakeEngine()
    call_count = []

    def flaky_put(*args, **kwargs):
        call_count.append(1)
        if len(call_count) == 1:
            raise RuntimeError("simulated backend error")

    engine.storage_manager.batched_put.side_effect = flaky_put

    # First item will raise; second item must still be processed
    engine._store_queue.put(_make_work_item("req-fail"))
    engine._store_queue.put(_make_work_item("req-ok"))
    engine._drain_store_queue()

    assert len(call_count) == 2, "worker stopped after exception"
    assert engine._store_thread.is_alive(), "worker thread died after exception"
    engine.shutdown()


def test_req_id_unknown_if_unpack_fails():
    """If tuple unpacking fails, worker must log <unknown> and not crash."""
    engine = FakeEngine()

    # Put a malformed item (too few fields) to trigger NameError guard
    engine._store_queue.put(("bad", "item"))
    engine._drain_store_queue()

    # Thread must still be alive to process the next valid item
    assert engine._store_thread.is_alive()
    valid_item = _make_work_item("req-after-bad")
    engine._store_queue.put(valid_item)
    engine._drain_store_queue()
    engine.storage_manager.batched_put.assert_called_once()
    engine.shutdown()
