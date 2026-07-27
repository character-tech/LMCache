# SPDX-License-Identifier: Apache-2.0
"""
Managing objects and memory for L1 cache
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal
import os
import queue
import threading
import time
import zlib

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import TTLLock
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1ManagerProtocol,
    L1MemoryManager,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)

# Debug-only instrumentation for the KV-cache stale-slot-reuse investigation:
# hashes CPU-side KV bytes on every store/retrieve and logs a WARNING on a
# retrieve-time mismatch. Off by default -- zero overhead unless enabled.
LMCACHE_HASH_CHECK_DEBUG = os.getenv("LMCACHE_HASH_CHECK_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)


@dataclass
class _HashCheckJob:
    """[LMCACHE_HASH_CHECK_DEBUG] One store or retrieve event queued for
    off-thread hashing.

    ``entry`` is the LIVE ``L1ObjectState``, not a copy -- safe because
    the caller keeps its write_lock/read_lock held until the worker
    releases it, so the slot can't be reused while queued. Avoids an
    ~18 MB memcpy under L1Manager's global lock.
    """

    key: ObjectKey
    entry: "L1ObjectState"
    is_store: bool
    # For retrieve jobs: how many read-lock counts to release (matches
    # finish_read's own `total = 1 + extra_count`).
    unlock_count: int
    data_ptr: int
    timestamp: float


class _HashCheckShard:
    """[LMCACHE_HASH_CHECK_DEBUG] One shard of the hash-check worker: one
    queue, one dedicated thread, one private ``_last_store`` table.

    Sharded by key (not a shared queue drained by N threads) so a key's
    store always hashes before that key's later retrieve, and each
    shard's ``_last_store`` needs no cross-thread locking.
    """

    # Bounds worst-case memory (~500 bytes/entry across all shards)
    # instead of growing unboundedly over a multi-hour run. FIFO
    # eviction just silently skips the check for a key that ages out
    # before its next retrieve.
    _MAX_TRACKED_KEYS_TOTAL = 1_000_000

    # How often (in processed jobs) to log a queue-wait summary -- lets
    # us see from a run's log whether this shard ever neared the TTL
    # boundary, even on a clean run where the skip guard never fires.
    _WAIT_LOG_INTERVAL = 2000

    def __init__(self, manager: "L1Manager", max_tracked_keys: int, index: int) -> None:
        self._manager = manager
        self._max_tracked_keys = max_tracked_keys
        self._index = index
        self._queue: "queue.Queue[_HashCheckJob | None]" = queue.Queue()
        # key -> (hash, data_ptr, store_time) for the most recent store.
        # Private to this shard -- no lock needed.
        self._last_store: "OrderedDict[ObjectKey, tuple[int, int, float]]" = (
            OrderedDict()
        )
        self._jobs_processed = 0
        self._max_wait_since_last_log = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name=f"lmcache-hash-check-debug-{index}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, job: _HashCheckJob) -> None:
        self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            wait_s = time.time() - job.timestamp
            if wait_s > self._max_wait_since_last_log:
                self._max_wait_since_last_log = wait_s
            self._jobs_processed += 1
            if self._jobs_processed % self._WAIT_LOG_INTERVAL == 0:
                logger.info(
                    "L1Manager: hash-check shard %d queue-wait summary: "
                    "jobs_processed=%d max_wait_s_this_window=%.3f "
                    "queue_depth_now=%d",
                    self._index,
                    self._jobs_processed,
                    self._max_wait_since_last_log,
                    self._queue.qsize(),
                )
                self._max_wait_since_last_log = 0.0
            try:
                # The per-key lock stays held while queued, but its TTL
                # (600s write / 300s read) expires on its own timer
                # regardless -- if this job waited that long, the slot
                # may have been legitimately reused, and hashing it
                # would be a false positive from this debug patch, not
                # the real bug. Two checks: is_locked() (False = our TTL
                # fired, nobody re-locked since) and data_ptr unchanged
                # (catches TTL-expiry + immediate re-lock landing before
                # we get here). Verified necessary: a run with only the
                # data_ptr check saw 100% of its mismatches at/above the
                # TTL boundary with the guard never firing.
                #
                # IMPORTANT LIMITATION (independent Opus code review,
                # 2026-07-27): this guard cannot distinguish "I skipped
                # because MY queueing was slow" from "this IS the real
                # bug" -- premature free + reuse (the actual mechanism
                # under investigation) produces the identical signature
                # (lock not held / data_ptr changed) that this guard
                # uses to skip. So a real corruption event can be
                # silently absorbed here instead of logged as a
                # MISMATCH. A logged HASH CHECK MISMATCH remains solid,
                # one-directional evidence the bug fired; a run with
                # zero mismatches or many TTL EXPIRY skips is NOT
                # evidence the bug didn't fire -- only that this run
                # didn't catch it. Accepted tradeoff, not fixed further.
                lock = job.entry.write_lock if job.is_store else job.entry.read_lock
                if not lock.is_locked() or job.entry.memory_obj.data_ptr != job.data_ptr:
                    logger.warning(
                        "L1Manager: HASH CHECK TTL EXPIRY (not a real "
                        "corruption finding -- debug-patch artifact, "
                        "comparison skipped) on %s for key %s: "
                        "lock_still_locked=%s queued_data_ptr=%s "
                        "current_data_ptr=%s is_store=%s queue_wait_s=%s",
                        "store" if job.is_store else "retrieve",
                        job.key,
                        lock.is_locked(),
                        job.data_ptr,
                        job.entry.memory_obj.data_ptr,
                        job.is_store,
                        time.time() - job.timestamp,
                    )
                    continue
                content_hash = zlib.crc32(job.entry.memory_obj.byte_array)
                if job.is_store:
                    self._last_store[job.key] = (
                        content_hash,
                        job.data_ptr,
                        job.timestamp,
                    )
                    self._last_store.move_to_end(job.key)
                    if len(self._last_store) > self._max_tracked_keys:
                        self._last_store.popitem(last=False)
                else:
                    prior = self._last_store.get(job.key)
                    if prior is not None:
                        store_hash, store_data_ptr, store_time = prior
                        if content_hash != store_hash:
                            same_ptr = job.data_ptr == store_data_ptr
                            logger.warning(
                                "L1Manager: HASH CHECK MISMATCH on retrieve "
                                "for key %s: store_hash=%s retrieve_hash=%s "
                                "store_time=%s retrieve_time=%s "
                                "elapsed_s=%s store_data_ptr=%s "
                                "retrieve_data_ptr=%s same_data_ptr=%s",
                                job.key,
                                store_hash,
                                content_hash,
                                store_time,
                                job.timestamp,
                                job.timestamp - store_time,
                                store_data_ptr,
                                job.data_ptr,
                                same_ptr,
                            )
            finally:
                # Always release, even if hashing/logging raised, so a
                # bug in this debug path can never leak a permanently
                # held lock on a real L1 slot.
                if job.is_store:
                    self._manager._hash_check_finish_write_unlock(job.key)
                else:
                    self._manager._hash_check_finish_read_unlock(
                        job.key, job.unlock_count
                    )


class _HashCheckWorker:
    """[LMCACHE_HASH_CHECK_DEBUG] Routes jobs to a fixed pool of
    ``_HashCheckShard`` instances, keyed by ``hash(ObjectKey)``, so a
    key's jobs always land on the same shard (preserving per-key
    ordering) while different keys drain in parallel. Sharding is
    necessary at real traffic volumes: a single-threaded version was
    tried and its queue grew unboundedly (100k+ jobs deep within an
    hour at 5 QPS), which itself produced TTL-expiry false positives
    -- see the TTL-expiry check above for how those are now caught
    regardless of shard count.
    """

    def __init__(self, manager: "L1Manager", num_shards: int) -> None:
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        max_tracked_keys = max(
            1, _HashCheckShard._MAX_TRACKED_KEYS_TOTAL // num_shards
        )
        self._shards = [
            _HashCheckShard(manager, max_tracked_keys, i) for i in range(num_shards)
        ]

    def submit(self, job: _HashCheckJob) -> None:
        shard = self._shards[hash(job.key) % len(self._shards)]
        shard.submit(job)


_hash_check_worker: "_HashCheckWorker | None" = None

# Number of shards (queue + thread + private _last_store table). A
# single shard's one thread cannot keep pace with real traffic (5 QPS x
# many KV chunks/request, each a zlib.crc32 over an ~18MB DeepSeek-V3
# MLA object) -- verified via queue-wait telemetry showing unbounded
# growth with 1 shard. More shards = more parallel drain throughput.
LMCACHE_HASH_CHECK_NUM_SHARDS = max(
    1, int(os.getenv("LMCACHE_HASH_CHECK_NUM_SHARDS", "8"))
)


def _get_hash_check_worker(manager: "L1Manager") -> _HashCheckWorker:
    global _hash_check_worker
    if _hash_check_worker is None:
        _hash_check_worker = _HashCheckWorker(manager, LMCACHE_HASH_CHECK_NUM_SHARDS)
    return _hash_check_worker


# Internal classes and helper functions
@dataclass
class L1ObjectState:
    """
    The internal state of an object in L1 cache
    """

    memory_obj: MemoryObj
    """ The memory object stored in L1 cache. """

    write_lock: TTLLock
    """ Whether the object is write-locked. """

    read_lock: TTLLock
    """ The read lock with TTL for the object. """

    is_temporary: bool
    """ Whether the object is temporary (need to be deleted after read). """

    def available_for_read(self) -> bool:
        """Check if the object is available for read.

        Returns:
            True if the object is not write-locked, False otherwise.
        """
        return not self.write_lock.is_locked()

    def available_for_write(self) -> bool:
        """Check if the object is available for write.

        Returns:
            True if the object is not write-locked and has no read locks
            and is not a temporary object, False otherwise.
        """

        return (
            not self.write_lock.is_locked()
            and not self.read_lock.is_locked()
            and not self.is_temporary
        )


def l1_mgr_synchronized(func):
    """
    Decorator to mark L1Manager methods as thread-safe
    """

    def wrapper(self: "L1Manager", *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


L1OperationResult = tuple[L1Error, MemoryObj | None]

# Upper bound for the count parameter in reserve_read / finish_read
# to prevent a single call from holding the global lock for too long.
MAX_READ_LOCK_COUNT = 128


def _validate_extra_count(extra_count: int) -> int:
    """Validate and clamp extra_count.

    Args:
        extra_count: Extra lock count on top of the
            default 1 lock.

    Returns:
        Clamped value in [0, MAX_READ_LOCK_COUNT - 1].
    """
    if extra_count < 0:
        logger.warning(
            "L1Manager: extra_count=%d is invalid, clamping to 0",
            extra_count,
        )
        return 0
    upper = MAX_READ_LOCK_COUNT - 1
    if extra_count > upper:
        logger.warning(
            "L1Manager: extra_count=%d exceeds limit=%d, clamping",
            extra_count,
            upper,
        )
        return upper
    return extra_count


def _l1_usage_ratio_or_zero(target: "L1Manager | None") -> float:
    """Return ``target.get_memory_usage()`` as a 0.0-1.0 ratio.

    Returns 0.0 when ``target`` is None or ``total_bytes`` is zero so the
    observable-gauge callback never raises during scrape.
    """
    if target is None:
        return 0.0
    used, total = target.get_memory_usage()
    if total <= 0:
        return 0.0
    return used / total


# Main classes


class L1Manager:
    """
    Object lifecycle state machine for L1 cache

          +--------+
          |  None  | <---------------------------------------+
          +--------+                                         |
            |   ^                                            |
            |   | (write lock expired)                       | delete()
            |   |                                            |
    reserve |   +----------------------+                     |
    write() |                          |                     |
            v                          |                     |
      +--------------+           +-----------+               |
      | write_locked |           |           |---------------+
      |              |---------->|   ready   |
      |              | finish_   |           |---------------+
      +--------------+ write()   +-----------+               |
            ^                          |                     |
            |                          | reserve_read()      | finish_read()
            +--------------------------+                     | (if count becomes 0)
                 reserve_write()       |                     |
                                       v                     |
                               +-----------------+           |
                               |   read_locked   |-----------+
                               |   (count = 1)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                               +-----------------+
                               |   read_locked   |
                               |   (count = 2)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                                   (...)  (...)
                               (Higher Counts)

    For every operation on list of keys, the operation is atomic
    """

    # Singleton dispatch for ``lmcache_mp.l1_memory_usage_bytes``: tests may
    # construct multiple L1Managers but the OTel SDK only honors the first
    # gauge registration, so the callback reads from the most recently built
    # instance via ``_gauge_target``.
    _gauge_registered: bool = False
    _gauge_target: "L1Manager | None" = None

    def __init__(self, config: L1ManagerConfig):
        self._lock = threading.Lock()

        self._objects: dict[ObjectKey, L1ObjectState] = {}

        # GDS and CPU L1 are mutually exclusive tiers, each driven by its own
        # config: the GDS tier reads only ``gds_l1_config`` (slab size +
        # alignment), the CPU tier only ``memory_config``.
        self._memory_manager: L1ManagerProtocol
        if config.gds_l1_config is not None:
            self._memory_manager = GDSL1MemoryManager(config.gds_l1_config)
            logger.info("L1Manager: GDS L1 tier enabled; CPU pinned-DRAM L1 disabled")
        else:
            self._memory_manager = L1MemoryManager(config.memory_config)

        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        self._registered_listeners: list[L1ManagerListener] = []

        self._event_bus = get_event_bus()

        L1Manager._gauge_target = self
        if not L1Manager._gauge_registered:
            L1Manager._gauge_registered = True
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_memory_usage_bytes",
                "Bytes currently held in L1 cache",
                lambda: (
                    L1Manager._gauge_target.get_memory_usage()[0]
                    if L1Manager._gauge_target is not None
                    else 0
                ),
            )
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_usage_ratio",
                "L1 used/total ratio (0.0–1.0)",
                lambda: _l1_usage_ratio_or_zero(L1Manager._gauge_target),
            )

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._registered_listeners.append(listener)

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys.

        Args:
            keys: The list of object keys to reserve
                read access for.
            extra_count: Extra read locks on top of the
                default 1 lock.  Total locks acquired per
                key = 1 + extra_count.  Useful when multiple
                workers each consume one read lock for the
                same key (e.g. MLA models with TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple
            of (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key exists but is not
                readable.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.available_for_read():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.lock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Unsafe read the read-locked objects without adding new read locks.

        This method does not acquire read locks. Therefore, the caller need
        to make sure the `unsafe_read` is called between `reserve_read` and
        `finish_read` calls.

        Args:
            keys: The list of object keys to read.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key is not readable (in this case, not read-locked).
        """
        ret: dict[ObjectKey, L1OperationResult] = {}

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.read_lock.is_locked():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        return ret

    @l1_mgr_synchronized
    def finish_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys.

        Will delete the object if it is temporary and read
        count reaches zero.

        Args:
            keys: The list of object keys to finish read
                access for.
            extra_count: Extra read locks to release on top
                of the default 1.  Must match the
                ``extra_count`` used in the corresponding
                ``reserve_read`` call.

        Returns:
            A dictionary mapping each object key to an
            L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is write-locked or
                non-read-locked, which means the reader may
                read inconsistent data.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                logger.warning(
                    "L1Manager: finish read on non-existing key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on write-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if not entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on non-read-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if LMCACHE_HASH_CHECK_DEBUG:
                # Defer the unlock + temp-object cleanup to
                # _hash_check_finish_read_unlock, called once
                # _HashCheckWorker finishes hashing off-thread.
                _get_hash_check_worker(self).submit(
                    _HashCheckJob(
                        key=key,
                        entry=entry,
                        is_store=False,
                        unlock_count=total,
                        data_ptr=entry.memory_obj.data_ptr,
                        timestamp=time.time(),
                    )
                )
                ret[key] = L1Error.SUCCESS
                successful_keys.append(key)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.unlock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.unlock()
            if entry.is_temporary and not entry.read_lock.is_locked():
                # NOTE: temporary objects shouldn't have write-locks
                need_to_free.append(entry.memory_obj)
                need_to_free_keys.append(key)
                del self._objects[key]

            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": need_to_free_keys},
            )
        )

        return ret

    @l1_mgr_synchronized
    def _hash_check_finish_read_unlock(
        self, key: "ObjectKey", unlock_count: int
    ) -> None:
        """[LMCACHE_HASH_CHECK_DEBUG] Completes the unlock + cleanup
        finish_read deferred for ``key`` while the worker hashed its
        buffer. Mirrors the non-debug path exactly."""
        entry = self._objects.get(key, None)
        if entry is None:
            # Deleted by something else in the meantime (e.g. an
            # explicit delete()) -- nothing left to unlock/clean up.
            return

        for _ in range(unlock_count):
            entry.read_lock.unlock()

        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        if entry.is_temporary and not entry.read_lock.is_locked():
            need_to_free.append(entry.memory_obj)
            need_to_free_keys.append(key)
            del self._objects[key]

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished([key])
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": [key]},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": need_to_free_keys},
            )
        )

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve write access for the given keys.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: The list of booleans indicating whether each key is
                temporary.
            shape_spec: The memory layout description for the objects to be
                allocated.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_WRITABLE: The key exists but is not writable.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.
        """
        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=False):
            entry = self._objects.get(key, None)
            if entry is None:
                need_to_allocate.append((key, is_temp))
                continue

            if mode == "new":
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            if not entry.available_for_write():
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            entry.write_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        # Early return if no allocation is needed
        if len(need_to_allocate) == 0:
            return ret

        # Don't allow allocation in "update" mode
        if mode == "update":
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
            return ret

        err, allocated_objs = self._memory_manager.allocate(
            layout_desc, len(need_to_allocate)
        )

        if err != L1Error.SUCCESS:
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)

            # Free the memory if partial allocation succeeded
            if allocated_objs:
                self._memory_manager.free(allocated_objs)

        else:
            for (key, is_temp), mem_obj in zip(
                need_to_allocate, allocated_objs, strict=False
            ):
                self._objects[key] = L1ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=is_temp,
                )
                self._objects[key].write_lock.lock()
                ret[key] = (L1Error.SUCCESS, mem_obj)
                successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys.

        Args:
            keys: The list of object keys to finish write access for.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it's read-locked,
                which means the writer may have caused inconsistent data.
        """
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on non-write-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on read-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if LMCACHE_HASH_CHECK_DEBUG:
                # Defer the unlock to _hash_check_finish_write_unlock,
                # called once the worker finishes hashing off-thread.
                # A same-key re-store just overwrites _last_store, so a
                # queued job never compares against a stale generation.
                _get_hash_check_worker(self).submit(
                    _HashCheckJob(
                        key=key,
                        entry=entry,
                        is_store=True,
                        unlock_count=0,
                        data_ptr=entry.memory_obj.data_ptr,
                        timestamp=time.time(),
                    )
                )
                ret[key] = L1Error.SUCCESS
                successful_keys.append(key)
                continue

            entry.write_lock.unlock()
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_write_finished(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def _hash_check_finish_write_unlock(self, key: "ObjectKey") -> None:
        """[LMCACHE_HASH_CHECK_DEBUG] Releases the write lock finish_write
        deferred for ``key`` while the worker hashed its buffer. Event
        publishing already happened synchronously in finish_write; only
        the lock release itself was deferred."""
        entry = self._objects.get(key, None)
        if entry is None:
            # Deleted by something else in the meantime -- nothing left
            # to unlock.
            return
        entry.write_lock.unlock()

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys.

        This is used by the prefetch controller after successfully loading
        data from L2 into write-reserved L1 buffers. It transitions the
        object from write-locked to read-locked in a single atomic step,
        preventing a race window where eviction could interfere.

        Args:
            keys: Keys to transition from write-locked to read-locked.
            extra_count: Extra read locks on top of the default 1 lock.
                Total locks acquired per key = 1 + extra_count.  Useful
                when multiple TP workers each consume one read lock for
                the same key (e.g. MLA models with TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it already
                has read locks.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on "
                    "non-write-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on read-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            entry.write_lock.unlock()
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache.

        Args:
            keys: The list of object keys to delete.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IS_LOCKED: The key is locked (either write-locked or read-locked
                and cannot be deleted).
        """
        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if entry.read_lock.is_locked() or entry.write_lock.is_locked():
                ret[key] = L1Error.KEY_IS_LOCKED
                continue

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def touch_keys(self, keys: list[ObjectKey]):
        """Touch the given keys, marking the keys as accessed(retrieved or stored).

        Args:
            keys: The list of object keys to touch.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)

    @l1_mgr_synchronized
    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        if force:
            logger.warning(
                "L1Manager: force-clearing all %d objects "
                "(including locked ones). This may corrupt in-flight "
                "store/prefetch operations — use with caution.",
                len(self._objects),
            )
            all_keys = list(self._objects.keys())
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(all_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": all_keys},
                )
            )
            logger.info(
                "L1Manager: cleared %d objects, 0 remaining.",
                len(all_keys),
            )
            return

        keys_to_clear: list[ObjectKey] = []
        objs_to_free: list[MemoryObj] = []
        locked_count = 0

        for key, entry in list(self._objects.items()):
            if entry.write_lock.is_locked() or entry.read_lock.is_locked():
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]

        self._memory_manager.free(objs_to_free)

        if keys_to_clear:
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(keys_to_clear)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": keys_to_clear},
                )
            )

        logger.info(
            "L1Manager: cleared %d objects, %d locked objects remaining.",
            len(keys_to_clear),
            locked_count,
        )

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Check if a key is eligible for eviction (not locked).

        This method does NOT acquire the global L1Manager lock.
        L1Manager.delete() will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key exists and is not locked (neither read-locked
            nor write-locked), False otherwise.
        """
        entry = self._objects.get(key, None)
        if entry is None:
            return False
        return not entry.read_lock.is_locked() and not entry.write_lock.is_locked()

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current memory usage of L1 cache.

        Returns:
            A tuple of (used_memory_bytes, total_memory_bytes).

        Note:
            In the future, we many want to make a "callback" based mechanism
            via "L1ManagerListener" to notify the memory usage changes.
        """
        return self._memory_manager.get_memory_usage()

    def get_l1_memory_desc(self):
        """Return an L1MemoryDesc describing the underlying L1 memory buffer."""
        return self._memory_manager.get_l1_memory_desc()

    def close(self) -> None:
        """Close the L1Manager and free all resources."""
        with self._lock:
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()

        self._memory_manager.close()

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state."""
        write_locked = 0
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.write_lock.is_locked():
                write_locked += 1
            if entry.read_lock.is_locked():
                read_locked += 1
            if entry.is_temporary:
                temporary += 1
        used, total = self._memory_manager.get_memory_usage()
        return {
            "is_healthy": self._memory_manager.memcheck(),
            "total_object_count": len(self._objects),
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "temporary_count": temporary,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    # Debugging APIs
    @l1_mgr_synchronized
    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Get the internal state of the object with the given key.

        Args:
            key: The object key.

        Returns:
            The L1ObjectState if the object exists, None otherwise.
        """
        return self._objects.get(key, None)

    @l1_mgr_synchronized
    def memcheck(self) -> bool:
        """Perform memory check for L1 cache."""
        mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        num_write_locked = 0
        num_read_locked = 0
        for key, entry in self._objects.items():
            if entry.write_lock.is_locked():
                num_write_locked += 1
            if entry.read_lock.is_locked():
                num_read_locked += 1

        logger.info(
            "L1Manager memcheck: total objects = %d, write-locked = %d, "
            "read-locked = %d",
            len(self._objects),
            num_write_locked,
            num_read_locked,
        )
        return mem_check_result
