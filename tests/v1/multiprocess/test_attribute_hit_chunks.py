# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure L1/L2 hit & serve attribution helper."""

# First Party
from lmcache.v1.multiprocess.modules.lookup import _attribute_hit_chunks

CHUNK = 256


def test_all_l1_resident_no_native():
    # 4 found chunks, keys 0..3 all resident, 1 key/chunk, no native hit.
    assert _attribute_hit_chunks(4, (0, 1, 2, 3), 0, CHUNK, 1) == (4, 0, 4, 0)


def test_all_l2_loaded_no_native():
    assert _attribute_hit_chunks(4, (), 0, CHUNK, 1) == (0, 4, 0, 4)


def test_mixed_residency():
    # chunks 0,1 resident; 2,3 loaded from L2.
    assert _attribute_hit_chunks(4, (0, 1), 0, CHUNK, 1) == (2, 2, 2, 2)


def test_native_trims_serve_but_not_hit():
    # 4 found chunks, first 2 resident; native covers 2 chunks (512 tokens).
    # Presence split unchanged; serve counts only chunks 2,3 (both L2).
    assert _attribute_hit_chunks(4, (0, 1), 2 * CHUNK, CHUNK, 1) == (2, 2, 0, 2)


def test_native_covers_everything():
    assert _attribute_hit_chunks(4, (0, 1, 2, 3), 100 * CHUNK, CHUNK, 1) == (
        4,
        0,
        0,
        0,
    )


def test_partially_resident_chunk_attributes_to_l2():
    # 2 keys per chunk; chunk 0 fully resident (keys 0,1), chunk 1 has
    # only key 2 -> partial -> attributes to L2.
    assert _attribute_hit_chunks(4, (0, 1, 2), 0, CHUNK, 2) == (1, 1, 1, 1)


def test_non_prefix_residency():
    # Residency need not be a prefix: chunks 0 and 2 resident, 1 loaded.
    assert _attribute_hit_chunks(3, (0, 2), 0, CHUNK, 1) == (2, 1, 2, 1)


def test_empty_and_degenerate():
    assert _attribute_hit_chunks(0, (), 0, CHUNK, 1) == (0, 0, 0, 0)
    assert _attribute_hit_chunks(4, (0,), 0, CHUNK, 0) == (0, 0, 0, 0)
    assert _attribute_hit_chunks(4, (0,), 0, 0, 1) == (0, 0, 0, 0)


def test_sub_chunk_native_rounds_down():
    # 100 native tokens < 1 chunk -> trims nothing.
    assert _attribute_hit_chunks(2, (0,), 100, CHUNK, 1) == (1, 1, 1, 1)
