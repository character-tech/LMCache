# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import replace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    BoundedBytesLimiter,
    RawBlockCore,
    RawBlockCoreConfig,
    encode_object_key,
    validate_raw_block_io_options,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def make_paced_config(path, **pacing) -> RawBlockCoreConfig:
    """Build a posix raw-block core config with I/O pacing knobs overridden."""
    return replace(make_raw_block_core_config(path), **pacing)


# ---------------------------------------------------------------------------
# BoundedBytesLimiter unit tests
# ---------------------------------------------------------------------------


def test_limiter_grants_up_to_capacity():
    limiter = BoundedBytesLimiter(1000)

    assert limiter.acquire(400) == 400
    assert limiter.acquire(600) == 600
    assert limiter.granted_bytes() == 1000
    assert limiter.available_bytes() == 0

    limiter.release(400)
    limiter.release(600)
    assert limiter.available_bytes() == 1000


def test_limiter_blocks_until_release():
    limiter = BoundedBytesLimiter(100)
    granted = limiter.acquire(100)
    assert granted == 100

    acquired = threading.Event()

    def wait_and_acquire():
        limiter.acquire(50)
        acquired.set()

    t = threading.Thread(target=wait_and_acquire)
    t.start()
    assert not acquired.wait(0.2), "acquire should block while window is full"
    limiter.release(granted)
    assert acquired.wait(5.0), "acquire should proceed after release"
    t.join(timeout=5.0)


def test_limiter_oversized_acquire_passes_alone():
    limiter = BoundedBytesLimiter(100)

    # An op larger than the window is granted the whole window (clamped) and
    # must not deadlock.
    granted = limiter.acquire(500)
    assert granted == 100
    assert limiter.granted_bytes() == 100

    limiter.release(granted)
    assert limiter.available_bytes() == 100


def test_limiter_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        BoundedBytesLimiter(0)
    with pytest.raises(ValueError):
        BoundedBytesLimiter(-1)


# ---------------------------------------------------------------------------
# Config parsing / validation
# ---------------------------------------------------------------------------


def test_validate_raw_block_io_options_pacing_knobs():
    validate_raw_block_io_options(
        iouring_queue_depth=8,
        max_inflight_read_bytes=0,
        max_inflight_write_bytes=0,
    )
    validate_raw_block_io_options(
        iouring_queue_depth=8,
        max_inflight_read_bytes=67108864,
        max_inflight_write_bytes=33554432,
    )
    with pytest.raises(ValueError):
        validate_raw_block_io_options(
            iouring_queue_depth=8,
            max_inflight_read_bytes=-1,
        )
    with pytest.raises(ValueError):
        validate_raw_block_io_options(
            iouring_queue_depth=8,
            max_inflight_write_bytes=-1,
        )


def test_raw_block_core_config_pacing_defaults_off():
    assert (
        RawBlockCoreConfig.__dataclass_fields__["max_inflight_read_bytes"].default == 0
    )
    assert (
        RawBlockCoreConfig.__dataclass_fields__["max_inflight_write_bytes"].default == 0
    )


# ---------------------------------------------------------------------------
# Concurrency pacing against a real temp-file core
# ---------------------------------------------------------------------------


def _install_instrumented_rawdev(core: RawBlockCore) -> dict:
    """Monkeypatch ``core._rawdev`` to track concurrent pread bytes.

    Returns a state dict with ``inflight`` and ``peak`` byte counters.
    """
    state = {"lock": threading.Lock(), "inflight": 0, "peak": 0}
    original_rawdev = core._rawdev()

    class Tracked:
        def __init__(self, inner):
            self._inner = inner

        def pread_into(self, offset, buf, payload_len, total_len=None):
            total = int(total_len if total_len is not None else payload_len)
            with state["lock"]:
                state["inflight"] += total
                state["peak"] = max(state["peak"], state["inflight"])
            try:
                return self._inner.pread_into(offset, buf, payload_len, total_len)
            finally:
                with state["lock"]:
                    state["inflight"] -= total

        def __getattr__(self, name):
            return getattr(self._inner, name)

    core._rawdev = lambda: Tracked(original_rawdev)  # type: ignore[method-assign]
    return state


def _populate_slots(core: RawBlockCore, num_slots: int, payload_size: int):
    """Store ``num_slots`` keys and return (specs, payloads)."""
    specs = [encode_object_key(make_object_key(i)) for i in range(num_slots)]
    payloads = [bytes([i % 251 + 1]) * payload_size for i in range(num_slots)]
    objects = [make_memory_obj(p) for p in payloads]
    result = core.put_many(specs, objects)
    assert all(result.results), "test setup: all puts must succeed"
    return specs, payloads


def _spawn_and_wait(targets) -> None:
    threads = [threading.Thread(target=fn) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60.0)
        assert not t.is_alive(), "worker thread hung under pacing"


def test_pacing_bounds_concurrent_read_bytes(tmp_path):
    path = make_raw_block_file(tmp_path)
    payload = 60 * 1024
    # Window fits exactly one payload-sized read.
    config = make_paced_config(path, max_inflight_read_bytes=payload)
    core = RawBlockCore(config, key_namespace="object")
    state = _install_instrumented_rawdev(core)

    try:
        specs, payloads = _populate_slots(core, num_slots=8, payload_size=payload)
        assert state["peak"] <= payload, "store path is not read-paced"

        def load_one(spec, size):
            obj = make_empty_memory_obj(size)
            results = core.load_many_into([spec.encoded], [obj])
            assert results == [True]

        before_peak = state["peak"]
        _spawn_and_wait([lambda s=spec: load_one(s, payload) for spec in specs])

        # Without pacing, 8 concurrent threads would read 8x payload bytes
        # concurrently; with the window, peak concurrent reads must stay
        # within one payload.
        assert state["peak"] - before_peak <= payload
        assert state["peak"] > 0
    finally:
        core.close()


def test_pacing_disabled_by_default_no_bound(tmp_path):
    path = make_raw_block_file(tmp_path)
    payload = 60 * 1024
    config = make_raw_block_core_config(path)  # pacing defaults off
    core = RawBlockCore(config, key_namespace="object")
    state = _install_instrumented_rawdev(core)

    try:
        specs, payloads = _populate_slots(core, num_slots=8, payload_size=payload)

        barrier = threading.Barrier(8)

        def load_one_slow(spec, size):
            obj = make_empty_memory_obj(size)
            barrier.wait()
            results = core.load_many_into([spec.encoded], [obj])
            assert results == [True]

        # Synchronize threads so their reads overlap despite fast temp-file
        # I/O; without a window, peak must be able to exceed one payload.
        threads = [
            threading.Thread(target=load_one_slow, args=(spec, payload))
            for spec in specs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60.0)
            assert not t.is_alive()

        assert state["peak"] > payload, "expected overlapping unpaced reads"
    finally:
        core.close()


def test_pacing_bounds_concurrent_write_bytes(tmp_path):
    path = make_raw_block_file(tmp_path)
    payload = 60 * 1024
    # Window fits roughly one payload write plus header.
    config = make_paced_config(path, max_inflight_write_bytes=payload + 4096)
    core = RawBlockCore(config, key_namespace="object")

    state = {"lock": threading.Lock(), "inflight": 0, "peak": 0}
    original_rawdev = core._rawdev()

    class TrackedWrite:
        def __init__(self, inner):
            self._inner = inner

        def pwrite_from_buffer(self, offset, buf, payload_len, total_len=None):
            total = int(total_len if total_len is not None else payload_len)
            with state["lock"]:
                state["inflight"] += total
                state["peak"] = max(state["peak"], state["inflight"])
            try:
                return self._inner.pwrite_from_buffer(
                    offset, buf, payload_len, total_len
                )
            finally:
                with state["lock"]:
                    state["inflight"] -= total

        def __getattr__(self, name):
            return getattr(self._inner, name)

    core._rawdev = lambda: TrackedWrite(original_rawdev)  # type: ignore[method-assign]

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(8)]
        payloads = [bytes([i + 1]) * payload for i in range(8)]
        objects = [make_memory_obj(p) for p in payloads]

        put_results: list[list[bool]] = []

        def put_slice(keys, objs):
            put_results.append(core.put_many(keys, objs).results)

        halves = len(specs) // 2
        _spawn_and_wait(
            [
                lambda: put_slice(specs[:halves], objects[:halves]),
                lambda: put_slice(specs[halves:], objects[halves:]),
            ]
        )
        assert all(all(r) for r in put_results)
        # Two concurrent put slices would overlap writes; the window keeps
        # concurrent in-flight write bytes within one payload + header.
        assert state["peak"] <= payload + 4096
        assert state["peak"] > 0
    finally:
        core.close()


def test_pacing_status_report(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_paced_config(
        path,
        max_inflight_read_bytes=4096,
        max_inflight_write_bytes=8192,
    )
    core = RawBlockCore(config, key_namespace="object")
    try:
        status = core.report_status()
        assert status["max_inflight_read_bytes"] == 4096
        assert status["max_inflight_write_bytes"] == 8192
        assert status["inflight_read_window_bytes"] == 0
        assert status["inflight_write_window_bytes"] == 0
    finally:
        core.close()
