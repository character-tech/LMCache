# SPDX-License-Identifier: Apache-2.0
"""Experimental CacheGen serde for heterogeneous Gemma 4 KV objects.

Gemma 4 serving objects can contain several KV tensor groups with different
head geometries.  The distributed serde interface receives those groups in a
single :class:`MemoryObj`, so the legacy homogeneous CacheGen serializer cannot
be applied to ``MemoryObj.tensor`` directly.  This module encodes each group
independently and stores the resulting payloads in one length-prefixed envelope.

The current MVP targets the FP8-byte layout used by the Character Gemma 4
deployment.  It is intentionally strict: unknown shapes and non-uint8 source
groups are rejected instead of being interpreted with a guessed layout.
"""

# Future
from __future__ import annotations

# Standard
from typing import Protocol
import math
import struct

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig,
    CacheGenGPUEncoderOutput,
    QuantizationSpec,
)
from lmcache.storage_backend.serde.cachegen_decoder import (
    decode_function_gpu,
    do_dequantize,
)
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj

_MAGIC = b"LMCG4V1\0"
_HEADER = struct.Struct("<8sI")
_LENGTH = struct.Struct("<Q")
_CACHEGEN_CDF_BINS = 33
_PICKLE_MARGIN_BYTES = 8 * 1024 * 1024


class Gemma4GroupCodec(Protocol):
    """Codec contract used by the heterogeneous object envelope."""

    def encode(self, group: torch.Tensor, head_dim: int) -> bytes:
        """Encode one uint8 FP8-bit group and return a CacheGen payload."""
        ...

    def decode(
        self,
        payload: bytes,
        shape: torch.Size,
        head_dim: int,
    ) -> torch.Tensor:
        """Decode one payload into a CPU uint8 tensor with ``shape``."""
        ...


def _validate_group_layout(
    shape: torch.Size,
    dtype: torch.dtype,
    head_dim_by_hidden_size: dict[int, int],
) -> int:
    if len(shape) != 4 or int(shape[0]) != 2:
        raise ValueError(
            "gemma4_cachegen requires group shape [2, layers, tokens, hidden], "
            f"got {tuple(shape)}"
        )
    if dtype != torch.uint8:
        raise ValueError(
            "gemma4_cachegen requires uint8 groups containing FP8 bit patterns, "
            f"got {dtype}"
        )
    hidden_size = int(shape[3])
    head_dim = head_dim_by_hidden_size.get(hidden_size)
    if head_dim is None or hidden_size % head_dim != 0:
        known = ", ".join(str(v) for v in sorted(head_dim_by_hidden_size))
        raise ValueError(
            f"gemma4_cachegen has no head geometry for hidden size {hidden_size}; "
            f"configured hidden sizes: {known}"
        )
    return head_dim


def _copy_bytes_to_tensor(payload: bytes, dst: torch.Tensor) -> None:
    if dst.device.type != "cpu":
        raise ValueError("gemma4_cachegen serialized buffers must be on CPU")
    if dst.dtype != torch.uint8:
        raise ValueError("gemma4_cachegen serialized buffers must use torch.uint8")
    if dst.numel() < len(payload):
        raise ValueError(
            f"gemma4_cachegen destination has {dst.numel()} bytes, needs {len(payload)}"
        )
    source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dst.flatten()[: len(payload)].copy_(source)


def _pack_payloads(payloads: list[bytes]) -> bytes:
    parts = [_HEADER.pack(_MAGIC, len(payloads))]
    for payload in payloads:
        parts.append(_LENGTH.pack(len(payload)))
        parts.append(payload)
    return b"".join(parts)


def _unpack_payloads(blob: bytes, expected_count: int) -> list[bytes]:
    if len(blob) < _HEADER.size:
        raise ValueError("gemma4_cachegen payload is shorter than its header")
    magic, count = _HEADER.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError("gemma4_cachegen payload has an invalid magic value")
    if count != expected_count:
        raise ValueError(
            f"gemma4_cachegen payload contains {count} groups, "
            f"destination expects {expected_count}"
        )

    offset = _HEADER.size
    payloads: list[bytes] = []
    for _ in range(count):
        if offset + _LENGTH.size > len(blob):
            raise ValueError("gemma4_cachegen payload has a truncated length field")
        (payload_size,) = _LENGTH.unpack_from(blob, offset)
        offset += _LENGTH.size
        end = offset + payload_size
        if end > len(blob):
            raise ValueError("gemma4_cachegen payload has a truncated group")
        payloads.append(blob[offset:end])
        offset = end
    if offset != len(blob):
        raise ValueError("gemma4_cachegen payload has trailing bytes")
    return payloads


class CacheGenFp8GroupCodec:
    """Run legacy CacheGen independently over one FP8-byte KV group.

    Args:
        bins: Uniform CacheGen quantization bins for both K and V.
        fp8_dtype: PyTorch FP8 dtype represented by the input uint8 bytes.
        staging_dtype: Floating-point dtype used between FP8 and CacheGen.

    Notes:
        CacheGen currently serializes its internal tensors with pickle.  This
        codec must therefore only read payloads produced by a trusted LMCache
        deployment, never arbitrary user input.
    """

    def __init__(
        self,
        bins: int = 28,
        fp8_dtype: torch.dtype | None = None,
        staging_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if fp8_dtype is None:
            fp8_dtype = getattr(torch, "float8_e4m3fnuz", None)
        if not isinstance(fp8_dtype, torch.dtype):
            raise ValueError("torch.float8_e4m3fnuz is unavailable")
        if bins < 4 or bins > 32 or bins % 2 != 0:
            raise ValueError("CacheGen bins must be an even integer in [4, 32]")
        if fp8_dtype.itemsize != 1:
            raise ValueError("gemma4_cachegen requires a one-byte FP8 dtype")
        if not staging_dtype.is_floating_point:
            raise ValueError("gemma4_cachegen staging_dtype must be floating point")
        self.bins = bins
        self.fp8_dtype = fp8_dtype
        self.staging_dtype = staging_dtype

    def encode(self, group: torch.Tensor, head_dim: int) -> bytes:
        """Encode one uint8 FP8-bit group and return a CacheGen payload.

        Args:
            group: CPU tensor shaped ``[2, layers, tokens, hidden]``.
            head_dim: Head dimension for this group.

        Returns:
            Self-contained legacy CacheGen payload.
        """
        if group.dtype != torch.uint8 or len(group.shape) != 4:
            raise ValueError("CacheGenFp8GroupCodec.encode received an invalid group")
        _, layers, tokens, hidden_size = (int(v) for v in group.shape)
        if hidden_size % head_dim != 0:
            raise ValueError("hidden size must be divisible by head_dim")
        num_heads = hidden_size // head_dim

        fp_group = group.view(self.fp8_dtype).to(
            device=torch_device_type,
            dtype=self.staging_dtype,
        )
        cachegen_input = (
            fp_group.reshape(2, layers, tokens, num_heads, head_dim)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        config = CacheGenConfig(
            nlayers=layers,
            kspecs=[QuantizationSpec(0, layers, self.bins)],
            vspecs=[QuantizationSpec(0, layers, self.bins)],
        )
        key_bins = torch.full(
            (layers,), float(self.bins), dtype=torch.float32, device=torch_device_type
        )
        value_bins = key_bins.clone()
        output = encode_function(
            cachegen_input,
            config,
            key_bins,
            value_bins,
            tokens,
        )
        return output.to_bytes()

    def decode(
        self,
        payload: bytes,
        shape: torch.Size,
        head_dim: int,
    ) -> torch.Tensor:
        """Decode one CacheGen payload into FP8 bytes on CPU.

        Args:
            payload: Trusted payload returned by :meth:`encode`.
            shape: Expected ``[2, layers, tokens, hidden]`` result shape.
            head_dim: Head dimension for this group.

        Returns:
            CPU uint8 tensor containing reconstructed FP8 bit patterns.
        """
        _, expected_layers, expected_tokens, hidden_size = (int(v) for v in shape)
        num_heads = hidden_size // head_dim
        encoded = CacheGenGPUEncoderOutput.from_bytes(payload)
        device = torch.device(torch_device_type)
        encoded.cdf = encoded.cdf.to(device)
        encoded.max_tensors_key = encoded.max_tensors_key.to(device)
        encoded.max_tensors_value = encoded.max_tensors_value.to(device)
        for chunk in encoded.data_chunks:
            chunk.bytestream = chunk.bytestream.to(device)
            chunk.bytestream_lengths = chunk.bytestream_lengths.to(device)

        layers = int(encoded.max_tensors_key.shape[0])
        tokens = int(encoded.max_tensors_key.shape[1])
        if (
            layers != expected_layers
            or tokens != expected_tokens
            or encoded.num_heads != num_heads
            or encoded.head_size != head_dim
        ):
            raise ValueError(
                "gemma4_cachegen payload geometry does not match destination: "
                f"payload=({layers}, {tokens}, {encoded.num_heads}, "
                f"{encoded.head_size}), expected=({expected_layers}, "
                f"{expected_tokens}, {num_heads}, {head_dim})"
            )

        channels = num_heads * head_dim
        output_buffer = torch.empty(
            (tokens, 2 * layers * channels), dtype=torch.uint8, device=device
        )
        key, value = decode_function_gpu(
            encoded.cdf,
            encoded.data_chunks,
            layers,
            tokens,
            output_buffer,
        )
        bins = torch.full(
            (layers,), float(self.bins), dtype=torch.float32, device=device
        )
        key = do_dequantize(key, bins, encoded.max_tensors_key)
        value = do_dequantize(value, bins, encoded.max_tensors_value)
        recovered = (
            torch.stack([key, value])
            .reshape(2, layers, tokens, num_heads, head_dim)
            .reshape(shape)
            .to(self.fp8_dtype)
            .view(torch.uint8)
        )
        return recovered.cpu()


class Gemma4CacheGenSerializer(Serializer):
    """Serialize all heterogeneous groups in one Gemma 4 KV object."""

    def __init__(
        self,
        codec: Gemma4GroupCodec,
        head_dim_by_hidden_size: dict[int, int],
    ) -> None:
        self.codec = codec
        self.head_dim_by_hidden_size = dict(head_dim_by_hidden_size)

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Encode every source group and write one envelope to ``dst``.

        Args:
            src: Heterogeneous Gemma 4 KV object.
            dst: CPU uint8 destination buffer.

        Returns:
            Number of serialized bytes written.
        """
        payloads: list[bytes] = []
        for index, (shape, dtype) in enumerate(
            zip(src.get_shapes(), src.get_dtypes(), strict=True)
        ):
            head_dim = _validate_group_layout(
                shape, dtype, self.head_dim_by_hidden_size
            )
            group = src.get_tensor(index)
            if group is None:
                raise ValueError(f"gemma4_cachegen source group {index} has no tensor")
            payloads.append(self.codec.encode(group, head_dim))

        dst_tensor = dst.tensor
        if dst_tensor is None:
            raise ValueError("gemma4_cachegen destination has no tensor")
        blob = _pack_payloads(payloads)
        _copy_bytes_to_tensor(blob, dst_tensor)
        return len(blob)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return a conservative upper bound for the CacheGen envelope.

        The bound includes one raw byte per quantized symbol, CacheGen's
        per-channel CDF, per-token maxima, bytestream lengths, envelope bytes,
        and an 8 MiB pickle/alignment margin.

        Args:
            layout_desc: Source heterogeneous KV layout.

        Returns:
            Estimated maximum serialized size in bytes.
        """
        estimate = _HEADER.size + _PICKLE_MARGIN_BYTES
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            _validate_group_layout(shape, dtype, self.head_dim_by_hidden_size)
            _, layers, tokens, hidden_size = (int(v) for v in shape)
            raw_symbol_bytes = math.prod(shape)
            cdf_bytes = 2 * layers * hidden_size * _CACHEGEN_CDF_BINS * 2
            maxima_bytes = 2 * layers * tokens * 4
            num_chunks = max(1, math.ceil(tokens / 256))
            length_bytes = num_chunks * 2 * layers * hidden_size * 4
            estimate += (
                _LENGTH.size
                + raw_symbol_bytes
                + cdf_bytes
                + maxima_bytes
                + length_bytes
            )
        return estimate


class Gemma4CacheGenDeserializer(Deserializer):
    """Restore all heterogeneous groups in one Gemma 4 KV object."""

    def __init__(
        self,
        codec: Gemma4GroupCodec,
        head_dim_by_hidden_size: dict[int, int],
    ) -> None:
        self.codec = codec
        self.head_dim_by_hidden_size = dict(head_dim_by_hidden_size)

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Decode a CacheGen envelope into the destination KV object.

        Args:
            src: CPU uint8 serialized buffer.
            dst: Heterogeneous Gemma 4 KV object.
        """
        src_tensor = src.tensor
        if src_tensor is None:
            raise ValueError("gemma4_cachegen source has no tensor")
        if src_tensor.device.type != "cpu" or src_tensor.dtype != torch.uint8:
            raise ValueError("gemma4_cachegen source must be a CPU uint8 buffer")
        blob = bytes(src_tensor.contiguous().numpy())
        shapes = dst.get_shapes()
        dtypes = dst.get_dtypes()
        payloads = _unpack_payloads(blob, len(shapes))

        for index, (payload, shape, dtype) in enumerate(
            zip(payloads, shapes, dtypes, strict=True)
        ):
            head_dim = _validate_group_layout(
                shape, dtype, self.head_dim_by_hidden_size
            )
            target = dst.get_tensor(index)
            if target is None:
                raise ValueError(
                    f"gemma4_cachegen destination group {index} has no tensor"
                )
            recovered = self.codec.decode(payload, shape, head_dim)
            if recovered.shape != target.shape or recovered.dtype != target.dtype:
                raise ValueError(
                    "gemma4_cachegen codec returned "
                    f"{recovered.shape}/{recovered.dtype}, "
                    f"expected {target.shape}/{target.dtype}"
                )
            target.copy_(recovered.to(target.device))


def _create_gemma4_cachegen_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    bins = int(kwargs.get("bins", 28))  # type: ignore[call-overload]
    fp8_name = str(kwargs.get("fp8_dtype", "float8_e4m3fnuz"))
    staging_name = str(kwargs.get("staging_dtype", "bfloat16"))
    fp8_dtype = getattr(torch, fp8_name, None)
    staging_dtype = getattr(torch, staging_name, None)
    if not isinstance(fp8_dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {fp8_name!r}")
    if not isinstance(staging_dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {staging_name!r}")

    raw_geometry = kwargs.get(
        "head_dim_by_hidden_size",
        {"4096": 256, "2048": 512},
    )
    if not isinstance(raw_geometry, dict):
        raise ValueError("head_dim_by_hidden_size must be a JSON object")
    geometry = {int(k): int(v) for k, v in raw_geometry.items()}
    codec = CacheGenFp8GroupCodec(
        bins=bins,
        fp8_dtype=fp8_dtype,
        staging_dtype=staging_dtype,
    )
    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        Gemma4CacheGenSerializer(codec, geometry),
        Gemma4CacheGenDeserializer(codec, geometry),
        max_workers=max_workers,
    )


register_serde_factory("gemma4_cachegen", _create_gemma4_cachegen_serde)
