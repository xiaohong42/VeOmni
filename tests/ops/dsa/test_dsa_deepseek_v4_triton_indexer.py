# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 Triton lighting indexer (ROCm) against the eager row."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from veomni.ops import resolve_op
from veomni.ops.platform import ROCM_GPU
from veomni.utils.device import get_device_type


DEVICE = get_device_type()
_ROCM_TRITON = ROCM_GPU.matches() and importlib.util.find_spec("triton") is not None
requires_rocm_triton = pytest.mark.skipif(not _ROCM_TRITON, reason="The Triton indexer row targets AMD ROCm GPUs")


def _inputs(seq_len, batch, heads, dim, compress, seed):
    torch.manual_seed(seed)
    q = torch.randn(seq_len, batch, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(seq_len // compress, batch, dim, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, batch, heads, device=DEVICE) * 0.05
    return q, k, weights


def _eager_and_triton():
    return (
        resolve_op("dsa_indexer", "deepseek_v4", "eager").wrapper,
        resolve_op("dsa_indexer", "deepseek_v4", "triton").wrapper,
    )


@pytest.mark.parametrize(
    ("heads", "dim", "q_dtype", "k_dtype", "match"),
    (
        (6, 128, torch.bfloat16, torch.bfloat16, "power-of-two head count"),
        (256, 128, torch.bfloat16, torch.bfloat16, "power-of-two head count"),
        (8, 96, torch.bfloat16, torch.bfloat16, "power-of-two head dim"),
        (8, 128, torch.float32, torch.float32, "bfloat16 or float16"),
        (8, 128, torch.bfloat16, torch.float16, "bfloat16 or float16"),
    ),
)
def test_triton_indexer_rejects_unsupported_inputs_before_launch(heads, dim, q_dtype, k_dtype, match):
    from veomni.ops.kernels.dsa.indexer.deepseek_v4 import triton as triton_indexer

    q = torch.zeros(8, 1, heads, dim, dtype=q_dtype)
    k = torch.zeros(2, 1, dim, dtype=k_dtype)
    with pytest.raises(ValueError, match=match):
        triton_indexer.wrapper(q, k, torch.zeros(8, 1, heads), 4, 2)


@requires_rocm_triton
@pytest.mark.parametrize(("heads", "seq_len", "topk"), ((8, 256, 16), (16, 200, 7), (64, 256, 64)))
def test_triton_indexer_matches_eager_causal(heads, seq_len, topk):
    """Causal compressed-KV selection, including rows that see fewer than topk entries."""
    eager, triton = _eager_and_triton()
    q, k, weights = _inputs(seq_len, 2, heads, 128, 4, seed=heads)

    expected_scores, expected_indices = eager(q, k, weights, 4, topk)
    actual_scores, actual_indices = triton(q, k, weights, 4, topk)

    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)
    assert (actual_indices == -1).any(), "the early rows must see fewer than topk entries"


@requires_rocm_triton
def test_triton_indexer_matches_eager_on_packed_ranges():
    """Packed segments restart their compressed range; rows outside every range select nothing."""
    eager, triton = _eager_and_triton()
    seq_len, compress, topk = 192, 4, 12
    q, k, weights = _inputs(seq_len, 1, 32, 128, compress, seed=7)
    starts, ends = [], []
    for begin, end in ((0, 80), (80, 192)):
        base = begin // compress
        for t in range(begin, end):
            starts.append(base)
            ends.append(base + (t - begin + 1) // compress)
    ks = torch.tensor(starts, device=DEVICE, dtype=torch.int32)
    ke = torch.tensor(ends, device=DEVICE, dtype=torch.int32)

    expected_scores, expected_indices = eager(q, k, weights, compress, topk, cu_seqlen_ks=ks, cu_seqlen_ke=ke)
    actual_scores, actual_indices = triton(q, k, weights, compress, topk, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)


@requires_rocm_triton
def test_triton_indexer_scores_supplied_indices_without_gradient():
    """Supplied indices are scored, not re-selected, and the scores leave autograd."""
    eager, triton = _eager_and_triton()
    q, k, weights = _inputs(128, 1, 16, 128, 4, seed=3)
    q.requires_grad_(True)
    indices = torch.randint(-1, 32, (1, 128, 9), device=DEVICE, dtype=torch.int32)

    expected_scores, _ = eager(q.detach(), k, weights, 4, 9, topk_indices=indices)
    actual_scores, actual_indices = triton(q, k, weights, 4, 9, topk_indices=indices)

    assert actual_indices is indices
    assert not actual_scores.requires_grad
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)


@requires_rocm_triton
def test_triton_indexer_matches_eager_at_v4_shape():
    """DeepSeek-V4's 64 index heads of dim 128 over a 4K-token sequence with compress ratio 4."""
    eager, triton = _eager_and_triton()
    q, k, weights = _inputs(4096, 1, 64, 128, 4, seed=11)

    expected_scores, expected_indices = eager(q, k, weights, 4, 512)
    actual_scores, actual_indices = triton(q, k, weights, 4, 512)

    # Only the head-sum order differs, so a pick may swap only between scores that agree to fp32 rounding.
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)
    assert (actual_indices == expected_indices).float().mean().item() > 0.999
