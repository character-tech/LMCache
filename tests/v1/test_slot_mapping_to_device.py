# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheConnectorV1Impl._slot_mapping_to_device and
_get_connector_stream, tested in isolation without vLLM or CUDA."""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch


def _load_methods():
    """Import the two unbound methods, skip if vLLM is not installed."""
    try:
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        return (
            LMCacheConnectorV1Impl._slot_mapping_to_device,
            LMCacheConnectorV1Impl._get_connector_stream,
        )
    except ImportError:
        pytest.skip("lmcache.integration.vllm not importable (vLLM not installed)")


def _make_stub(device_type="cpu", buf_size=64):
    """Return a minimal stub that exposes the same attrs as LMCacheConnectorV1Impl."""
    _smd, _gcs = _load_methods()

    stub = types.SimpleNamespace()
    stub.device = torch.device(device_type)
    stub._slot_mapping_pinned_buf = torch.zeros(buf_size, dtype=torch.long)
    stub._slot_mapping_buf_cursor = 0
    stub.lmcache_engine = None

    stub._slot_mapping_to_device = types.MethodType(_smd, stub)
    stub._get_connector_stream = types.MethodType(_gcs, stub)
    return stub


# ---------------------------------------------------------------------------
# _slot_mapping_to_device tests
# ---------------------------------------------------------------------------


def test_already_on_device():
    """slot_mapping already on target device → returned unchanged, cursor stays 0."""
    stub = _make_stub(device_type="cpu")
    sm = torch.tensor([0, 1, 2], dtype=torch.long)  # CPU tensor, device == "cpu"
    result = stub._slot_mapping_to_device(sm, kind="store")
    assert result is sm
    assert stub._slot_mapping_buf_cursor == 0


def _run_with_fake_cuda(stub, sm, kind="store"):
    """Drive _slot_mapping_to_device with CUDA ops patched out."""
    fake_stream = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=None)
    ctx_mgr.__exit__ = MagicMock(return_value=False)

    original_to = torch.Tensor.to

    def fake_to(self, device, non_blocking=False):
        # Simulate H2D by returning a copy on same device (avoids real CUDA).
        return self.clone()

    torch.Tensor.to = fake_to
    try:
        with (
            patch("torch.cuda.stream", return_value=ctx_mgr),
            patch("torch.cuda.current_stream", return_value=fake_stream),
        ):
            return stub._slot_mapping_to_device(sm, kind=kind)
    finally:
        torch.Tensor.to = original_to


def _cuda_typed_sm(values):
    """Build a CPU tensor whose .device.type reports 'cuda' via MagicMock."""
    real_tensor = torch.tensor(values, dtype=torch.long)
    sm = MagicMock()
    sm.device.type = "cuda"  # triggers the H2D branch
    sm.reshape.return_value = real_tensor
    sm.shape = real_tensor.shape
    return sm


def test_cpu_input_fits_buffer():
    """Tensor needing H2D: cursor advances by n, returned values are correct."""
    stub = _make_stub(device_type="cpu", buf_size=64)
    # Pretend stub.device is cuda so device.type mismatch triggers H2D path.
    stub.device = MagicMock()
    stub.device.type = "cuda"

    sm = _cuda_typed_sm([10, 20, 30])
    # sm.device.type == stub.device.type == "cuda" → fast-path. We need mismatch.
    # Make stub.device.type "rocm" to force the copy branch.
    stub.device.type = "rocm"

    result = _run_with_fake_cuda(stub, sm, kind="store")

    assert stub._slot_mapping_buf_cursor == 3
    assert list(result.numpy()) == [10, 20, 30]


def test_cpu_input_buffer_grows():
    """When slot_mapping exceeds buffer capacity, buffer grows and cursor is correct."""
    stub = _make_stub(device_type="cpu", buf_size=2)
    stub.device = MagicMock()
    stub.device.type = "rocm"  # force H2D branch

    sm = _cuda_typed_sm([1, 2, 3, 4, 5])

    _run_with_fake_cuda(stub, sm, kind="store")

    assert stub._slot_mapping_pinned_buf.numel() >= 5
    assert stub._slot_mapping_buf_cursor == 5


def test_cursor_advances_across_calls():
    """Two successive calls use non-overlapping buffer regions (cursor accumulates)."""
    stub = _make_stub(device_type="cpu", buf_size=20)
    stub.device = MagicMock()
    stub.device.type = "rocm"

    _run_with_fake_cuda(stub, _cuda_typed_sm([1, 2, 3]), kind="store")
    assert stub._slot_mapping_buf_cursor == 3

    _run_with_fake_cuda(stub, _cuda_typed_sm([4, 5]), kind="store")
    assert stub._slot_mapping_buf_cursor == 5


def test_buffer_wraps_when_cursor_exhausts():
    """When cursor + n would overflow but total buf is large enough, buffer grows."""
    stub = _make_stub(device_type="cpu", buf_size=4)
    stub.device = MagicMock()
    stub.device.type = "rocm"
    stub._slot_mapping_buf_cursor = 3  # only 1 slot left

    sm = _cuda_typed_sm([1, 2, 3])  # needs 3 → triggers growth

    _run_with_fake_cuda(stub, sm, kind="store")

    assert stub._slot_mapping_pinned_buf.numel() >= 3
    # After growth cursor resets to 0 then advances by 3.
    assert stub._slot_mapping_buf_cursor == 3


# ---------------------------------------------------------------------------
# _get_connector_stream tests
# ---------------------------------------------------------------------------


def test_get_connector_stream_none_engine():
    """lmcache_engine is None → fall back to current_stream."""
    stub = _make_stub()
    stub.lmcache_engine = None

    fake_stream = MagicMock()
    with patch("torch.cuda.current_stream", return_value=fake_stream):
        result = stub._get_connector_stream("store")

    assert result is fake_stream


def test_get_connector_stream_missing_attr():
    """gpu_connector exists but lacks the stream attr → fall back to current_stream."""
    stub = _make_stub()
    mock_engine = MagicMock(spec=[])  # no attributes by default
    # Give it gpu_connector but no store_stream on it.
    mock_engine.gpu_connector = MagicMock(spec=[])
    stub.lmcache_engine = mock_engine

    fake_stream = MagicMock()
    with patch("torch.cuda.current_stream", return_value=fake_stream):
        result = stub._get_connector_stream("store")

    assert result is fake_stream


def test_get_connector_stream_returns_correct_stream():
    """gpu_connector exposes the stream → return it directly."""
    stub = _make_stub()
    expected_stream = MagicMock()
    mock_connector = MagicMock()
    mock_connector.store_stream = expected_stream
    mock_engine = MagicMock()
    mock_engine.gpu_connector = mock_connector
    stub.lmcache_engine = mock_engine

    result = stub._get_connector_stream("store")

    assert result is expected_stream
