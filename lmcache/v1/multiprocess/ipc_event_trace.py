# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tracing for interprocess CUDA/HIP events.

Wraps ``torch.cuda.Event(interprocess=True)`` and
``torch.cuda.Event.from_ipc_handle`` so we can correlate destructor wall-time
with the count of live IPC events in the process. Used to test the hypothesis
that the HIP driver's ``SyncAllStreams`` (called from ``IPCEvent::~IPCEvent``)
takes longer as the number of mapped IPC handles grows, and that the resulting
window is what enables the GIL × CUDA-driver-lock AB-BA deadlock to manifest
at TP>=4 on AMD.

Activated by ``LMCACHE_DEBUG_IPC_LIFECYCLE=1``. No-op otherwise — callers
always invoke the same helpers, and the helpers fall back to the raw
``torch_dev.Event`` API when the env var is unset.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import itertools
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

# Mirror lmcache's torch_dev detection without depending on a symbol that may
# not exist in older lmcache installs. We only need Event() and from_ipc_handle.
if torch.cuda.is_available():
    torch_dev = torch.cuda
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    torch_dev = torch.xpu
elif hasattr(torch, "hpu") and torch.hpu.is_available():
    torch_dev = torch.hpu
else:
    torch_dev = torch.cuda  # last resort; tracing is a no-op without GPU anyway

logger = init_logger(__name__)

_ENABLED = os.environ.get("LMCACHE_DEBUG_IPC_LIFECYCLE", "0") == "1"
_id_counter = itertools.count()
_live_count = 0
_live_lock = threading.Lock()

# Pending host-callback counter. Incremented when a Python host callback is
# scheduled via launch_host_func; decremented when the callback actually runs
# on the CUDA/HIP driver thread. If destructors are wedging the driver lock,
# this counter stays high while live_count is also nonzero — the direct
# fingerprint of the GIL x driver-lock AB-BA.
_pending_callbacks = 0
_pending_lock = threading.Lock()
_cb_scheduled_total = 0
_cb_run_total = 0


def _bump(delta: int) -> int:
    global _live_count
    with _live_lock:
        _live_count += delta
        return _live_count


class _TrackedIPCEvent:
    """Wraps a torch interprocess Event to log create/destroy timing.

    Forwards attribute access for everything else (record, wait, ipc_handle).
    """

    __slots__ = ("_event", "_id", "_role", "_device", "_created_ns")

    def __init__(self, event: Any, role: str, device: Any) -> None:
        self._event = event
        self._id = next(_id_counter)
        self._role = role
        self._device = str(device)
        self._created_ns = time.monotonic_ns()
        live = _bump(+1)
        logger.info(
            "ipc_event create id=%d role=%s tid=%d device=%s live=%d",
            self._id,
            role,
            threading.get_ident(),
            self._device,
            live,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._event, name)

    def __del__(self) -> None:
        entry_ns = time.monotonic_ns()
        live_at_entry = _bump(0)  # snapshot without changing
        try:
            del self._event  # forces underlying torch Event destructor
        finally:
            exit_ns = time.monotonic_ns()
            _bump(-1)
            dt_ms = (exit_ns - entry_ns) / 1e6
            age_ms = (entry_ns - self._created_ns) / 1e6
            logger.info(
                "ipc_event destroy id=%d role=%s tid=%d device=%s "
                "live=%d dt_ms=%.3f age_ms=%.3f",
                self._id,
                self._role,
                threading.get_ident(),
                self._device,
                live_at_entry,
                dt_ms,
                age_ms,
            )


def new_event(role: str, device: Any) -> Any:
    """Construct ``torch_dev.Event(interprocess=True)``, wrapped if tracing
    is enabled. Returns the raw event otherwise."""
    raw = torch_dev.Event(interprocess=True)
    if not _ENABLED:
        return raw
    return _TrackedIPCEvent(raw, role, device)


def from_ipc_handle(device: Any, handle: bytes, role: str) -> Any:
    raw = torch_dev.Event.from_ipc_handle(device, handle)
    if not _ENABLED:
        return raw
    return _TrackedIPCEvent(raw, role, device)


def is_enabled() -> bool:
    return _ENABLED


def live_count() -> int:
    return _bump(0)


def _bump_pending(delta: int, scheduled: bool = False, ran: bool = False) -> int:
    global _pending_callbacks, _cb_scheduled_total, _cb_run_total
    with _pending_lock:
        _pending_callbacks += delta
        if scheduled:
            _cb_scheduled_total += 1
        if ran:
            _cb_run_total += 1
        return _pending_callbacks


def schedule_traced_callback(
    stream: Any, handler: Any, payload: Any, kind: str
) -> None:
    """Schedule ``handler(payload)`` on *stream* via launch_host_func, with
    pending-callback accounting. When the native recorder is not used (i.e.
    we're on the unpatched code path), this is the only signal that lets us
    correlate a wedged destructor with a stalled host callback."""
    if not _ENABLED:
        stream.launch_host_func(handler, payload)
        return

    seq = next(_id_counter)
    scheduled_ns = time.monotonic_ns()
    pending_after_schedule = _bump_pending(+1, scheduled=True)
    logger.info(
        "host_cb schedule id=%d kind=%s tid=%d pending=%d",
        seq,
        kind,
        threading.get_ident(),
        pending_after_schedule,
    )

    def _wrapped(p):
        run_ns = time.monotonic_ns()
        delay_ms = (run_ns - scheduled_ns) / 1e6
        try:
            handler(p)
        finally:
            pending_after_run = _bump_pending(-1, ran=True)
            logger.info(
                "host_cb run id=%d kind=%s tid=%d pending=%d delay_ms=%.3f",
                seq,
                kind,
                threading.get_ident(),
                pending_after_run,
                delay_ms,
            )

    stream.launch_host_func(_wrapped, payload)


def pending_callbacks() -> int:
    return _bump_pending(0)


def callback_totals() -> tuple[int, int]:
    with _pending_lock:
        return _cb_scheduled_total, _cb_run_total


def start_sampler(interval_seconds: float = 0.1) -> None:
    """Start a background thread that periodically logs live_count and
    pending_callbacks. Cheap (no GIL contention in the sampler itself
    beyond the log call) and runs forever. Idempotent."""
    if not _ENABLED:
        return
    if getattr(start_sampler, "_started", False):
        return
    start_sampler._started = True  # type: ignore[attr-defined]

    def _run():
        last_scheduled = 0
        last_run = 0
        while True:
            time.sleep(interval_seconds)
            sched, ran = callback_totals()
            d_sched = sched - last_scheduled
            d_run = ran - last_run
            last_scheduled, last_run = sched, ran
            logger.info(
                "sample live=%d pending_cb=%d d_scheduled=%d d_run=%d "
                "total_scheduled=%d total_run=%d",
                live_count(),
                pending_callbacks(),
                d_sched,
                d_run,
                sched,
                ran,
            )

    t = threading.Thread(target=_run, daemon=True, name="ipc-event-sampler")
    t.start()
