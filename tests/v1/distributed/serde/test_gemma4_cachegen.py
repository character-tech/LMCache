# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the heterogeneous Gemma 4 CacheGen envelope."""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import (
    Gemma4CacheGenDeserializer,
    Gemma4CacheGenSerializer,
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


class _RawGroupCodec:
    def encode(self, group: torch.Tensor, head_dim: int) -> bytes:
        del head_dim
        return bytes(group.contiguous().numpy())

    def decode(
        self,
        payload: bytes,
        shape: torch.Size,
        head_dim: int,
    ) -> torch.Tensor:
        del head_dim
        return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(shape)


def _source() -> _FakeMemoryObj:
    groups = [
        torch.arange(2 * 2 * 4 * 4096, dtype=torch.int64)
        .to(torch.uint8)
        .reshape(2, 2, 4, 4096),
        torch.arange(2 * 1 * 2 * 2048, dtype=torch.int64)
        .to(torch.uint8)
        .reshape(2, 1, 2, 2048),
    ]
    return _FakeMemoryObj(
        tensors=groups,
        shapes=[group.shape for group in groups],
        dtypes=[group.dtype for group in groups],
    )


def test_gemma4_cachegen_is_registered_by_default() -> None:
    """The experimental serde is reachable from L2 adapter JSON config."""
    assert "gemma4_cachegen" in get_registered_serde_types()


def test_heterogeneous_envelope_roundtrip() -> None:
    """Each heterogeneous group is preserved as a separate payload."""
    geometry = {4096: 256, 2048: 512}
    source = _source()
    serializer = Gemma4CacheGenSerializer(_RawGroupCodec(), geometry)
    layout = MemoryLayoutDesc(shapes=source.shapes, dtypes=source.dtypes)
    serialized = torch.zeros(
        serializer.estimate_serialized_size(layout), dtype=torch.uint8
    )
    destination_buffer = _FakeMemoryObj([], [], [], tensor=serialized)

    used = serializer.serialize(source, destination_buffer)  # type: ignore[arg-type]
    serialized_source = _FakeMemoryObj(
        [], [], [], tensor=serialized[:used].contiguous()
    )
    restored_groups = [torch.zeros_like(group) for group in source.tensors]
    destination = _FakeMemoryObj(
        restored_groups,
        source.shapes,
        source.dtypes,
    )
    Gemma4CacheGenDeserializer(_RawGroupCodec(), geometry).deserialize(  # type: ignore[arg-type]
        serialized_source,
        destination,  # type: ignore[arg-type]
    )

    for original, restored in zip(source.tensors, restored_groups, strict=True):
        assert torch.equal(original, restored)


def test_unknown_hidden_size_is_rejected() -> None:
    """Unknown Gemma head geometry fails instead of being guessed."""
    source = _FakeMemoryObj(
        tensors=[torch.zeros((2, 1, 2, 3072), dtype=torch.uint8)],
        shapes=[torch.Size((2, 1, 2, 3072))],
        dtypes=[torch.uint8],
    )
    serializer = Gemma4CacheGenSerializer(_RawGroupCodec(), {4096: 256})
    with pytest.raises(ValueError, match="no head geometry"):
        serializer.estimate_serialized_size(
            MemoryLayoutDesc(shapes=source.shapes, dtypes=source.dtypes)
        )


def test_non_uint8_group_is_rejected() -> None:
    """The MVP never silently reinterprets non-FP8 storage."""
    serializer = Gemma4CacheGenSerializer(_RawGroupCodec(), {4096: 256})
    with pytest.raises(ValueError, match="requires uint8"):
        serializer.estimate_serialized_size(
            MemoryLayoutDesc(
                shapes=[torch.Size((2, 1, 2, 4096))],
                dtypes=[torch.bfloat16],
            )
        )
