# SPDX-License-Identifier: Apache-2.0
"""Tests for blockwise INT4 compression of FP8 L2 KV objects."""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import (
    Fp8Int4L2Deserializer,
    Fp8Int4L2Serializer,
    get_registered_serde_types,
)


@dataclass
class _FakeMemoryObj:
    tensors: list[torch.Tensor]
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    tensor: Optional[torch.Tensor] = None

    def get_shapes(self) -> list[torch.Size]:
        return self.shapes

    def get_dtypes(self) -> list[torch.dtype]:
        return self.dtypes

    def get_tensor(self, index: int) -> torch.Tensor:
        return self.tensors[index]


def _fp8_group(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    values = torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.5
    return values.to(torch.float8_e4m3fnuz).view(torch.uint8)


def test_fp8_int4_l2_is_registered_by_default() -> None:
    """The codec can be selected from an L2 adapter serde config."""
    assert "fp8_int4_l2" in get_registered_serde_types()


def test_estimate_is_exact_and_close_to_two_to_one() -> None:
    """The estimate includes packed values, scales, and envelope bytes."""
    shapes = [torch.Size((2, 3, 8, 256)), torch.Size((2, 1, 4, 512))]
    layout = MemoryLayoutDesc(shapes=shapes, dtypes=[torch.uint8, torch.uint8])
    serializer = Fp8Int4L2Serializer(block_size=128, compute_device="cpu")
    estimated = serializer.estimate_serialized_size(layout)
    raw_bytes = sum(shape.numel() for shape in shapes)
    assert 1.9 < raw_bytes / estimated < 2.0


def test_fp8_scale_metadata_improves_small_block_ratio() -> None:
    """FP8 scales recover metadata capacity for finer quantization blocks."""
    shape = torch.Size((2, 3, 8, 256))
    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[torch.uint8])
    fp16_scales = Fp8Int4L2Serializer(
        block_size=16,
        scale_dtype=torch.float16,
        compute_device="cpu",
    ).estimate_serialized_size(layout)
    fp8_scales = Fp8Int4L2Serializer(
        block_size=16,
        scale_dtype=torch.float8_e4m3fnuz,
        compute_device="cpu",
    ).estimate_serialized_size(layout)
    assert fp8_scales < fp16_scales


def test_heterogeneous_roundtrip_has_bounded_error() -> None:
    """Heterogeneous FP8 groups round-trip with expected INT4 error."""
    groups = [
        _fp8_group((2, 3, 8, 256), 1),
        _fp8_group((2, 1, 4, 512), 2),
    ]
    shapes = [group.shape for group in groups]
    dtypes = [group.dtype for group in groups]
    source = _FakeMemoryObj(groups, shapes, dtypes)
    layout = MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)
    serializer = Fp8Int4L2Serializer(block_size=128, compute_device="cpu")
    serialized = torch.zeros(
        serializer.estimate_serialized_size(layout), dtype=torch.uint8
    )
    serialized_obj = _FakeMemoryObj([], [], [], tensor=serialized)

    used = serializer.serialize(source, serialized_obj)  # type: ignore[arg-type]
    assert used == serialized.numel()

    restored = [torch.zeros_like(group) for group in groups]
    restored_obj = _FakeMemoryObj(restored, shapes, dtypes)
    Fp8Int4L2Deserializer(  # type: ignore[arg-type]
        block_size=128, compute_device="cpu"
    ).deserialize(
        serialized_obj,
        restored_obj,  # type: ignore[arg-type]
    )

    for original_bits, restored_bits in zip(groups, restored, strict=True):
        original = original_bits.view(torch.float8_e4m3fnuz).float()
        recovered = restored_bits.view(torch.float8_e4m3fnuz).float()
        relative_l2 = torch.linalg.vector_norm(
            recovered - original
        ) / torch.linalg.vector_norm(original)
        assert relative_l2.item() < 0.13


def test_zero_groups_roundtrip_exactly() -> None:
    """All-zero blocks do not create zero scales or nonzero output."""
    group = torch.zeros((2, 1, 3, 65), dtype=torch.float8_e4m3fnuz).view(torch.uint8)
    source = _FakeMemoryObj([group], [group.shape], [group.dtype])
    layout = MemoryLayoutDesc(shapes=[group.shape], dtypes=[group.dtype])
    serializer = Fp8Int4L2Serializer(block_size=128, compute_device="cpu")
    serialized = torch.zeros(
        serializer.estimate_serialized_size(layout), dtype=torch.uint8
    )
    serialized_obj = _FakeMemoryObj([], [], [], tensor=serialized)
    serializer.serialize(source, serialized_obj)  # type: ignore[arg-type]

    restored = torch.empty_like(group)
    destination = _FakeMemoryObj([restored], [group.shape], [group.dtype])
    Fp8Int4L2Deserializer(  # type: ignore[arg-type]
        block_size=128, compute_device="cpu"
    ).deserialize(
        serialized_obj,
        destination,  # type: ignore[arg-type]
    )
    assert torch.equal(group, restored)


def test_non_uint8_layout_is_rejected() -> None:
    """The codec never guesses whether a wider tensor contains FP8 bits."""
    serializer = Fp8Int4L2Serializer(block_size=128, compute_device="cpu")
    with pytest.raises(ValueError, match="requires uint8"):
        serializer.estimate_serialized_size(
            MemoryLayoutDesc(
                shapes=[torch.Size((2, 1, 8, 256))],
                dtypes=[torch.bfloat16],
            )
        )


def test_invalid_block_size_is_rejected() -> None:
    """Nibble packing requires a positive even block size."""
    with pytest.raises(ValueError, match="positive even"):
        Fp8Int4L2Serializer(block_size=127)
