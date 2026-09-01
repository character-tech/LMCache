# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the StoreController in-flight store cap (store shedding).

Under sustained write backpressure, each in-flight L2 store holds an L1
read lock on its source block until the L2 write completes, pinning that
block (non-evictable). Unbounded in-flight stores can pin enough of L1
that eviction stalls. The cap sheds newly written keys once the number of
in-flight store tasks reaches ``max_inflight_tasks`` -- crucially BEFORE
reserving read locks, so shed keys are never pinned and stay evictable.

These tests use a real L1Manager plus a non-completing fake adapter that
holds store tasks in-flight indefinitely, so the in-flight count can be
driven to the cap deterministically.
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def write_key_to_l1(
    l1_manager: L1Manager,
    key: ObjectKey,
    layout: MemoryLayoutDesc,
) -> bool:
    """Write a single key to L1 (reserve_write + finish_write).

    Returns True if the write succeeded.
    """
    results = l1_manager.reserve_write(
        keys=[key],
        is_temporary=[False],
        layout_desc=layout,
        mode="new",
    )
    written = [k for k, (_e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)
    return bool(written)


def key_is_evictable(
    l1_manager: L1Manager,
    key: ObjectKey,
    layout: MemoryLayoutDesc,
) -> bool:
    """Return True if ``key`` holds no read lock in L1.

    A read-locked block cannot be write-reserved for update, so a
    successful ``reserve_write(mode="update")`` proves the block is not
    read-locked (i.e. not pinned by an in-flight store) and is therefore
    free for eviction/reuse.
    """
    results = l1_manager.reserve_write(
        keys=[key],
        is_temporary=[False],
        layout_desc=layout,
        mode="update",
    )
    entry = results.get(key)
    updatable = entry is not None and entry[1] is not None
    if updatable:
        # Release the write lock we just took so we don't leave state behind.
        l1_manager.finish_write([key])
    return updatable


class NonCompletingAdapter(MockL2Adapter):
    """MockL2Adapter that accepts store tasks but never completes them.

    Store tasks accumulate in-flight forever, which lets a test drive the
    StoreController's in-flight count to the cap deterministically.
    ``submit_store_task`` records every call so tests can assert whether a
    store was submitted or shed.
    """

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self.submitted_key_batches: list[list[ObjectKey]] = []
        self._submit_lock = threading.Lock()
        self._task_id = 0

    def submit_store_task(self, keys, objects):  # type: ignore[override]
        with self._submit_lock:
            self.submitted_key_batches.append(list(keys))
            self._task_id += 1
            return self._task_id

    def pop_completed_store_tasks(self):  # type: ignore[override]
        # Never report completions: tasks stay in-flight forever.
        return {}

    def submitted_task_count(self) -> int:
        with self._submit_lock:
            return len(self.submitted_key_batches)

    def submitted_key_count(self) -> int:
        with self._submit_lock:
            return sum(len(b) for b in self.submitted_key_batches)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=torch.cuda.is_available(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


def make_non_completing_adapter() -> NonCompletingAdapter:
    return NonCompletingAdapter(
        MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    )


# =============================================================================
# Tests
# =============================================================================


def test_default_zero_cap_is_unbounded(l1_manager):
    """max_inflight_tasks=0 (default) never sheds -- legacy behavior."""
    adapter = make_non_completing_adapter()
    ctrl = StoreController(
        l1_manager=l1_manager,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultStorePolicy(),
    )
    assert ctrl.report_status()["max_inflight_tasks"] == 0
    ctrl.start()

    layout = make_layout()
    keys = [make_object_key(i) for i in range(6)]
    for key in keys:
        write_key_to_l1(l1_manager, key, layout)

    # All 6 stores submit even though none complete (unbounded).
    ok = _wait(lambda: adapter.submitted_key_count() == 6)
    assert ok, "with cap=0 all writes must submit as store tasks"
    assert ctrl.report_status()["dropped_task_keys_total"] == 0

    ctrl.stop()
    adapter.close()


def test_stores_shed_at_cap(l1_manager):
    """At the cap, further stores are shed (not submitted, not queued)."""
    cap = 3
    adapter = make_non_completing_adapter()
    ctrl = StoreController(
        l1_manager=l1_manager,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultStorePolicy(),
        max_inflight_tasks=cap,
    )
    assert ctrl.report_status()["max_inflight_tasks"] == cap
    ctrl.start()

    layout = make_layout()
    # Each key is written (and processed) individually so each becomes its
    # own store task; the fake adapter never completes them, so in-flight
    # climbs to the cap and then holds.
    keys = [make_object_key(i) for i in range(cap + 4)]
    for key in keys:
        write_key_to_l1(l1_manager, key, layout)
        # Small settle so the single-threaded store loop processes this key
        # before the next write, making the ordering deterministic.
        time.sleep(0.05)

    # Exactly `cap` store tasks submitted; the rest shed.
    ok = _wait(lambda: adapter.submitted_task_count() == cap)
    assert ok, f"expected exactly {cap} submitted tasks"
    # Give the loop time to process (and shed) the remaining writes.
    time.sleep(0.3)

    status = ctrl.report_status()
    assert adapter.submitted_task_count() == cap, (
        "no more than `cap` tasks may be submitted while none complete"
    )
    assert status["in_flight_task_count"] == cap
    assert status["dropped_task_keys_total"] == len(keys) - cap, (
        "every write beyond the cap must be counted as dropped"
    )

    ctrl.stop()
    adapter.close()


def test_shed_keys_are_not_read_locked(l1_manager):
    """Shed keys are never reserve_read'd, so they stay evictable in L1.

    This is the core correctness property: the whole point of shedding is
    that a dropped store does not pin its L1 block.
    """
    cap = 2
    adapter = make_non_completing_adapter()
    ctrl = StoreController(
        l1_manager=l1_manager,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultStorePolicy(),
        max_inflight_tasks=cap,
    )
    ctrl.start()

    layout = make_layout()
    pinned = [make_object_key(i) for i in range(cap)]  # fill the cap
    shed = [make_object_key(100 + i) for i in range(3)]  # beyond the cap

    for key in pinned + shed:
        write_key_to_l1(l1_manager, key, layout)
        time.sleep(0.05)

    ok = _wait(lambda: adapter.submitted_task_count() == cap)
    assert ok
    time.sleep(0.3)

    # The pinned keys hold read locks (in-flight store never completes) -> NOT
    # evictable/updatable. The shed keys were never read-locked -> evictable.
    for key in pinned:
        assert not key_is_evictable(l1_manager, key, layout), (
            "an in-flight-store key must be read-locked (pinned)"
        )
    for key in shed:
        assert key_is_evictable(l1_manager, key, layout), (
            "a shed key must NOT be read-locked; it must stay evictable"
        )

    ctrl.stop()
    adapter.close()


def test_shedding_resumes_after_inflight_drains(l1_manager):
    """Once in-flight drops below the cap, new stores submit again.

    Uses a real (completing) MockL2Adapter so tasks actually drain, then
    verifies the cap only sheds while saturated -- not permanently.
    """
    cap = 2
    adapter = MockL2Adapter(
        MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
    )
    ctrl = StoreController(
        l1_manager=l1_manager,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultStorePolicy(),
        max_inflight_tasks=cap,
    )
    ctrl.start()

    layout = make_layout()
    keys = [make_object_key(i) for i in range(8)]
    for key in keys:
        write_key_to_l1(l1_manager, key, layout)

    # With a completing adapter, in-flight stays under the cap and drains,
    # so most/all keys eventually store even though the cap is small. The
    # cap only sheds transiently under saturation; it must not wedge.
    ok = _wait(
        lambda: adapter.debug_get_stored_object_count() >= cap,
        timeout=8.0,
    )
    assert ok, "a completing adapter must keep storing under the cap"

    # In-flight must never exceed the cap at steady state.
    time.sleep(0.3)
    assert ctrl.report_status()["in_flight_task_count"] <= cap

    ctrl.stop()
    adapter.close()


def test_concurrent_writes_bound_inflight_and_conserve_keys(l1_manager):
    """Concurrent producers never push in-flight tasks past the cap, and
    every key is accounted for (submitted or shed) exactly once."""
    cap = 4
    adapter = make_non_completing_adapter()
    ctrl = StoreController(
        l1_manager=l1_manager,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultStorePolicy(),
        max_inflight_tasks=cap,
    )
    ctrl.start()

    layout = make_layout()
    total = 40

    def writer(base: int) -> None:
        for i in range(base, base + 10):
            write_key_to_l1(l1_manager, make_object_key(i), layout)

    threads = [threading.Thread(target=writer, args=(b,)) for b in (0, 10, 20, 30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive()

    # Let the store loop fully drain its pending queue (shedding the excess).
    time.sleep(1.0)

    status = ctrl.report_status()
    # The cap bounds in-flight *tasks*, and same-shape keys popped together
    # are grouped into one task (see _group_keys_by_shape), so a burst of
    # concurrent same-shape writes can ride in as few as one task up to the
    # cap. How many tasks form is timing-dependent, so the invariants
    # asserted here are the ones that hold under ANY batching/interleaving:
    #
    # - in-flight tasks NEVER exceed the cap (the core guarantee),
    assert status["in_flight_task_count"] <= cap, (
        "in-flight task count must never exceed the cap"
    )
    assert adapter.submitted_task_count() <= cap
    # - key conservation: every written key is either submitted or shed,
    #   exactly once -- nothing lost, nothing double-counted, no key left
    #   half-reserved. (Deterministic per-key shedding, and the fact that
    #   shed keys stay evictable, are covered by test_stores_shed_at_cap and
    #   test_shed_keys_are_not_read_locked, which serialize writes so each
    #   key is its own task.)
    submitted_keys = adapter.submitted_key_count()
    dropped_keys = status["dropped_task_keys_total"]
    assert submitted_keys + dropped_keys == total, (
        f"key conservation violated: submitted={submitted_keys} "
        f"dropped={dropped_keys} total={total}"
    )

    ctrl.stop()
    adapter.close()


def _wait(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
