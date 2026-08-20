# SPDX-License-Identifier: Apache-2.0
"""Blockwise INT4 compression for FP8 KV objects stored in L2.

The serving engine continues to use its native FP8 KV representation.  This
serde only changes the bytes persisted by an LMCache L2 adapter: FP8 values are
converted to symmetric INT4 with one FP16 scale per block, packed two values per
byte, and restored to FP8 after an L2 hit.

This codec is lossy.  It is intentionally simple so its size, error, and
latency can be evaluated before investing in a fused GPU implementation.
"""

# Future
from __future__ import annotations

# Standard
import math
import struct

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj

_MAGIC = b"LMF8I4V2"
_HEADER = struct.Struct("<8sIII")
_GROUP_HEADER = struct.Struct("<QQQ")
_INT4_MAX = 7.0
_SCALE_DTYPE_CODES = {
    torch.float16: 1,
    torch.float8_e4m3fnuz: 2,
    torch.float8_e5m2fnuz: 3,
}


def _validate_block_size(block_size: int) -> None:
    if block_size < 2 or block_size % 2 != 0:
        raise ValueError("fp8_int4_l2 block_size must be a positive even integer")


def _validate_layout(shape: torch.Size, dtype: torch.dtype) -> None:
    if dtype != torch.uint8:
        raise ValueError(
            "fp8_int4_l2 requires uint8 tensors containing FP8 bit patterns, "
            f"got {dtype}"
        )
    if math.prod(shape) == 0:
        raise ValueError("fp8_int4_l2 does not support empty tensor groups")


def _scale_dtype_code(scale_dtype: torch.dtype) -> int:
    code = _SCALE_DTYPE_CODES.get(scale_dtype)
    if code is None:
        supported = ", ".join(str(dtype) for dtype in _SCALE_DTYPE_CODES)
        raise ValueError(
            f"fp8_int4_l2 unsupported scale dtype {scale_dtype}; supported: {supported}"
        )
    return code


def _validate_serialized_buffer(dst: torch.Tensor) -> None:
    if dst.device.type != "cpu" or dst.dtype != torch.uint8:
        raise ValueError("fp8_int4_l2 serialized buffers must be CPU uint8 tensors")


def _write_header(payload: bytes, dst: torch.Tensor, offset: int) -> int:
    end = offset + len(payload)
    if end > dst.numel():
        raise ValueError("fp8_int4_l2 destination is too small for its header")
    source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dst[offset:end].copy_(source)
    return end


def _encode_group(
    group: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    block_size: int,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if group.dtype != torch.uint8:
        raise ValueError("fp8_int4_l2 source groups must use torch.uint8")

    # Transfer the one-byte FP8 representation before expanding to float32.
    # Casting on CPU first would move four times as many bytes over PCIe.
    non_blocking = group.device.type == "cpu" and group.is_pinned()
    fp8_bits = group.contiguous().to(compute_device, non_blocking=non_blocking)
    values = fp8_bits.view(fp8_dtype).float().flatten()
    numel = values.numel()
    num_blocks = math.ceil(numel / block_size)
    padded_numel = num_blocks * block_size
    if padded_numel != numel:
        padded = torch.zeros(padded_numel, dtype=torch.float32, device=values.device)
        padded[:numel].copy_(values)
        values = padded

    blocks = values.reshape(num_blocks, block_size)
    maxima = blocks.abs().amax(dim=1)
    scales = torch.where(maxima == 0, torch.ones_like(maxima), maxima / _INT4_MAX)
    scales = scales.clamp_min(torch.finfo(scale_dtype).tiny)
    stored_scales = scales.to(scale_dtype)
    if not torch.isfinite(stored_scales.float()).all():
        raise ValueError("fp8_int4_l2 encountered non-finite KV values")

    # Quantize with the stored FP16 scale so decode uses exactly the same value.
    quantized = torch.round(blocks / stored_scales.float().unsqueeze(1))
    quantized = quantized.clamp(-7, 7).to(torch.int8)
    unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8).flatten()
    packed = unsigned[0::2] | (unsigned[1::2] << 4)

    scale_bytes = stored_scales.contiguous().view(torch.uint8)
    packed_bytes = packed.contiguous()
    return scale_bytes, packed_bytes, numel


def _decode_group(
    scale_payload: torch.Tensor,
    packed_payload: torch.Tensor,
    numel: int,
    shape: torch.Size,
    fp8_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    block_size: int,
    compute_device: torch.device,
) -> torch.Tensor:
    num_blocks = math.ceil(numel / block_size)
    expected_scale_bytes = num_blocks * scale_dtype.itemsize
    expected_packed_bytes = num_blocks * block_size // 2
    if scale_payload.numel() != expected_scale_bytes:
        raise ValueError(
            "fp8_int4_l2 scale payload has invalid size: "
            f"got {scale_payload.numel()}, expected {expected_scale_bytes}"
        )
    if packed_payload.numel() != expected_packed_bytes:
        raise ValueError(
            "fp8_int4_l2 packed payload has invalid size: "
            f"got {packed_payload.numel()}, expected {expected_packed_bytes}"
        )

    non_blocking = scale_payload.is_pinned() and packed_payload.is_pinned()
    scales = scale_payload.view(scale_dtype).to(
        device=compute_device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    packed = packed_payload.to(compute_device, non_blocking=non_blocking)
    unsigned = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=compute_device)
    unsigned[0::2] = packed & 0x0F
    unsigned[1::2] = packed >> 4
    quantized = (unsigned.to(torch.int16) - 8).reshape(num_blocks, block_size)
    values = quantized.float() * scales.unsqueeze(1)
    return values.flatten()[:numel].to(fp8_dtype).view(torch.uint8).reshape(shape)


class Fp8Int4L2Serializer(Serializer):
    """Compress FP8-byte KV tensor groups to blockwise symmetric INT4.

    Args:
        fp8_dtype: FP8 dtype represented by the source uint8 bit patterns.
        scale_dtype: Storage dtype for one positive scale per INT4 block.
        block_size: Number of consecutive KV values sharing one FP16 scale.
        compute_device: Device used for quantization. Defaults to LMCache's
            configured accelerator device.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fnuz,
        scale_dtype: torch.dtype = torch.float16,
        block_size: int = 128,
        compute_device: str | torch.device = torch_device_type,
    ) -> None:
        _validate_block_size(block_size)
        if fp8_dtype.itemsize != 1:
            raise ValueError("fp8_int4_l2 requires a one-byte FP8 dtype")
        self.scale_dtype_code = _scale_dtype_code(scale_dtype)
        self.fp8_dtype = fp8_dtype
        self.scale_dtype = scale_dtype
        self.block_size = block_size
        self.compute_device = torch.device(compute_device)

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Compress all FP8 source groups into one versioned byte envelope.

        Args:
            src: Source object containing uint8 tensors with FP8 bit patterns.
            dst: CPU uint8 buffer sized by :meth:`estimate_serialized_size`.

        Returns:
            Number of bytes written to ``dst``.
        """
        shapes = src.get_shapes()
        dtypes = src.get_dtypes()
        dst_tensor = dst.tensor
        if dst_tensor is None:
            raise ValueError("fp8_int4_l2 destination has no tensor")
        _validate_serialized_buffer(dst_tensor)
        dst_flat = dst_tensor.flatten()
        required = self.estimate_serialized_size(
            MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)
        )
        if dst_flat.numel() < required:
            raise ValueError(
                f"fp8_int4_l2 destination has {dst_flat.numel()} bytes, "
                f"needs {required}"
            )

        offset = _write_header(
            _HEADER.pack(
                _MAGIC,
                len(shapes),
                self.block_size,
                self.scale_dtype_code,
            ),
            dst_flat,
            0,
        )
        for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
            _validate_layout(shape, dtype)
            group = src.get_tensor(index)
            if group is None:
                raise ValueError(f"fp8_int4_l2 source group {index} has no tensor")
            scale_payload, packed_payload, numel = _encode_group(
                group,
                self.fp8_dtype,
                self.scale_dtype,
                self.block_size,
                self.compute_device,
            )
            offset = _write_header(
                _GROUP_HEADER.pack(
                    numel, scale_payload.numel(), packed_payload.numel()
                ),
                dst_flat,
                offset,
            )
            scale_end = offset + scale_payload.numel()
            packed_end = scale_end + packed_payload.numel()
            non_blocking = dst_flat.is_pinned()
            dst_flat[offset:scale_end].copy_(scale_payload, non_blocking=non_blocking)
            dst_flat[scale_end:packed_end].copy_(
                packed_payload, non_blocking=non_blocking
            )
            offset = packed_end
        if dst_flat.is_pinned() and self.compute_device.type == "cuda":
            torch.cuda.current_stream(self.compute_device).synchronize()
        return offset

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return the exact serialized size for the provided FP8 layout.

        Args:
            layout_desc: Shapes and dtypes of all source tensor groups.

        Returns:
            Envelope, FP16 scale, and packed INT4 bytes.
        """
        total = _HEADER.size
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            _validate_layout(shape, dtype)
            numel = math.prod(shape)
            num_blocks = math.ceil(numel / self.block_size)
            total += _GROUP_HEADER.size
            total += num_blocks * self.scale_dtype.itemsize
            total += num_blocks * self.block_size // 2
        return total


class Fp8Int4L2Deserializer(Deserializer):
    """Restore blockwise INT4 L2 payloads to FP8-byte KV tensor groups.

    Args:
        fp8_dtype: FP8 dtype represented by destination uint8 bit patterns.
        scale_dtype: Storage dtype expected for one scale per INT4 block.
        block_size: Expected INT4 quantization block size.
        compute_device: Device used for dequantization. Defaults to LMCache's
            configured accelerator device.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fnuz,
        scale_dtype: torch.dtype = torch.float16,
        block_size: int = 128,
        compute_device: str | torch.device = torch_device_type,
    ) -> None:
        _validate_block_size(block_size)
        if fp8_dtype.itemsize != 1:
            raise ValueError("fp8_int4_l2 requires a one-byte FP8 dtype")
        self.scale_dtype_code = _scale_dtype_code(scale_dtype)
        self.fp8_dtype = fp8_dtype
        self.scale_dtype = scale_dtype
        self.block_size = block_size
        self.compute_device = torch.device(compute_device)

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Decode an INT4 envelope into destination FP8-byte groups.

        Args:
            src: CPU uint8 serialized buffer.
            dst: Destination object with the original heterogeneous layout.
        """
        src_tensor = src.tensor
        if src_tensor is None:
            raise ValueError("fp8_int4_l2 source has no tensor")
        if src_tensor.device.type != "cpu" or src_tensor.dtype != torch.uint8:
            raise ValueError("fp8_int4_l2 source must be a CPU uint8 buffer")
        src_flat = src_tensor.flatten()
        payload_view = memoryview(src_flat.numpy())
        if src_flat.numel() < _HEADER.size:
            raise ValueError("fp8_int4_l2 payload is shorter than its header")
        magic, group_count, encoded_block_size, encoded_scale_dtype = (
            _HEADER.unpack_from(payload_view)
        )
        if magic != _MAGIC:
            raise ValueError("fp8_int4_l2 payload has an invalid magic value")
        if encoded_block_size != self.block_size:
            raise ValueError(
                "fp8_int4_l2 block size mismatch: "
                f"payload={encoded_block_size}, configured={self.block_size}"
            )
        if encoded_scale_dtype != self.scale_dtype_code:
            raise ValueError(
                "fp8_int4_l2 scale dtype mismatch: "
                f"payload={encoded_scale_dtype}, configured={self.scale_dtype_code}"
            )

        shapes = dst.get_shapes()
        dtypes = dst.get_dtypes()
        if group_count != len(shapes):
            raise ValueError(
                f"fp8_int4_l2 payload contains {group_count} groups, "
                f"destination expects {len(shapes)}"
            )

        offset = _HEADER.size
        has_async_copy = False
        for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
            _validate_layout(shape, dtype)
            if offset + _GROUP_HEADER.size > src_flat.numel():
                raise ValueError("fp8_int4_l2 payload has a truncated group header")
            numel, scale_bytes, packed_bytes = _GROUP_HEADER.unpack_from(
                payload_view, offset
            )
            offset += _GROUP_HEADER.size
            expected_numel = math.prod(shape)
            if numel != expected_numel:
                raise ValueError(
                    "fp8_int4_l2 group size mismatch: "
                    f"payload={numel}, destination={expected_numel}"
                )
            scale_end = offset + scale_bytes
            packed_end = scale_end + packed_bytes
            if packed_end > src_flat.numel():
                raise ValueError("fp8_int4_l2 payload has a truncated group")

            recovered = _decode_group(
                src_flat[offset:scale_end],
                src_flat[scale_end:packed_end],
                numel,
                shape,
                self.fp8_dtype,
                self.scale_dtype,
                self.block_size,
                self.compute_device,
            )
            target = dst.get_tensor(index)
            if target is None:
                raise ValueError(f"fp8_int4_l2 destination group {index} has no tensor")
            non_blocking = target.device.type == "cpu" and target.is_pinned()
            target.copy_(recovered, non_blocking=non_blocking)
            has_async_copy = has_async_copy or non_blocking
            offset = packed_end

        if offset != src_flat.numel():
            raise ValueError("fp8_int4_l2 payload has trailing bytes")
        if has_async_copy and self.compute_device.type == "cuda":
            torch.cuda.current_stream(self.compute_device).synchronize()


def _create_fp8_int4_l2_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    fp8_name = str(kwargs.get("fp8_dtype", "float8_e4m3fnuz"))
    fp8_dtype = getattr(torch, fp8_name, None)
    if not isinstance(fp8_dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {fp8_name!r}")
    scale_name = str(kwargs.get("scale_dtype", "float16"))
    scale_dtype = getattr(torch, scale_name, None)
    if not isinstance(scale_dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {scale_name!r}")
    block_size = int(kwargs.get("block_size", 128))  # type: ignore[call-overload]
    compute_device = str(kwargs.get("compute_device", torch_device_type))
    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        Fp8Int4L2Serializer(
            fp8_dtype=fp8_dtype,
            scale_dtype=scale_dtype,
            block_size=block_size,
            compute_device=compute_device,
        ),
        Fp8Int4L2Deserializer(
            fp8_dtype=fp8_dtype,
            scale_dtype=scale_dtype,
            block_size=block_size,
            compute_device=compute_device,
        ),
        max_workers=max_workers,
    )


register_serde_factory("fp8_int4_l2", _create_fp8_int4_l2_serde)
