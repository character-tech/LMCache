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

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("LMCACHE_DEBUG_IPC_LIFECYCLE", "0") == "1"
_id_counter = itertools.count()
_live_count = 0
_live_lock = threading.Lock()


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
