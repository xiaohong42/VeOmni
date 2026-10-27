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

"""DeepSeek-V4 lighting indexer in Triton, forward only.

The logits are the TileLang kernel's: ``sum_h w[s, h] * relu(q[s, h] . k[t])`` with
bf16 / fp16 products accumulated in fp32, written only inside each query's
``[cu_seqlen_ks, cu_seqlen_ke)`` range and ``-inf`` elsewhere. Every program
holds ``128 // heads`` queries with all their heads (128 rows of Q) and walks the
compressed-KV blocks that any of those queries may see, as the TileLang kernel
does. Only the head-sum order differs from eager / TileLang, so logits agree to
fp32 rounding and top-k picks can only differ between near-equal scores.

There is no backward: the returned scores carry no gradient. The indexer loss,
the only objective that differentiates them, requires the TileLang indexer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .....triton_cache import cached_triton_kernel


# Rows of Q (queries x heads) per program, the TileLang kernel's tile height.
_Q_ROWS = 128
_BLOCK_N = 64
# On gfx942 a single-buffered K tile is ~30% faster than Triton's default two stages.
_NUM_WARPS = 4
_NUM_STAGES = 1


@cached_triton_kernel
def _indexer_logits_kernel():
    """Compile and cache the Triton DeepSeek-V4 indexer logits kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        Q,
        K,
        W,
        KS,
        KE,
        LOGITS,
        seq_len,
        seq_len_kv,
        stride_ks,
        stride_ls,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = pid * BLOCK_M * HEADS + tl.arange(0, BLOCK_M * HEADS)
        offs_d = tl.arange(0, DIM)
        m_mask = offs_m < seq_len
        row_mask = rows < seq_len * HEADS
        # Q is [seq_len * heads, dim] and W [seq_len * heads]: row = query * heads + head.
        q = tl.load(Q + rows[:, None] * DIM + offs_d[None, :], mask=row_mask[:, None], other=0.0)
        w = tl.load(W + rows, mask=row_mask, other=0.0)
        ks = tl.load(KS + offs_m, mask=m_mask, other=seq_len_kv)
        ke = tl.load(KE + offs_m, mask=m_mask, other=0)
        lo = tl.maximum(tl.min(ks, axis=0), 0)
        hi = tl.minimum(tl.max(ke, axis=0), seq_len_kv)
        for start in range(lo // BLOCK_N * BLOCK_N, hi, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len_kv
            k = tl.load(K + offs_n[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k))
            scores = tl.maximum(scores, 0.0) * w[:, None]
            logits = tl.sum(tl.reshape(scores, (BLOCK_M, HEADS, BLOCK_N)), axis=1)
            valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
            logits = tl.where(valid, logits, float("-inf"))
            tl.store(
                LOGITS + offs_m[:, None] * stride_ls + offs_n[None, :],
                logits,
                mask=m_mask[:, None] & n_mask[None, :],
            )

    return kernel


def _causal_ranges(seq_len: int, compress_ratio: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Query row ``t`` sees compressed entries ``[0, (t + 1) // compress_ratio)``."""
    ends = (torch.arange(1, seq_len + 1, device=device, dtype=torch.int32) // compress_ratio).to(torch.int32)
    return torch.zeros_like(ends), ends


def indexer_logits(q: Tensor, k: Tensor, weights: Tensor, cu_seqlen_ks: Tensor, cu_seqlen_ke: Tensor) -> Tensor:
    """Return ``[S, S_kv]`` fp32 logits for one batch element.

    ``q`` is ``[S, H, D]``, ``k`` is ``[S_kv, D]`` (bf16 or fp16), ``weights`` is ``[S, H]``,
    and ``cu_seqlen_ks`` / ``cu_seqlen_ke`` are ``[S]`` int32 ranges of visible entries.
    """
    seq_len, heads, dim = q.shape
    seq_len_kv = k.shape[0]
    logits = torch.full((seq_len, seq_len_kv), float("-inf"), device=q.device, dtype=torch.float32)
    if seq_len == 0 or seq_len_kv == 0:
        return logits
    block_m = _Q_ROWS // heads
    _indexer_logits_kernel()[(-(-seq_len // block_m),)](
        q.contiguous().view(seq_len * heads, dim),
        k.contiguous(),
        weights.float().contiguous().view(seq_len * heads),
        cu_seqlen_ks.to(torch.int32).contiguous(),
        cu_seqlen_ke.to(torch.int32).contiguous(),
        logits,
        seq_len,
        seq_len_kv,
        dim,
        logits.stride(0),
        HEADS=heads,
        DIM=dim,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return logits


def _gather_scores(logits: Tensor, topk_indices: Tensor) -> Tensor:
    """Gather scores at ``topk_indices``. Invalid ids become ``-inf``."""
    if logits.shape[-1] == 0:
        return logits.new_full(topk_indices.shape, float("-inf"))
    valid = (topk_indices >= 0) & (topk_indices < logits.shape[-1])
    safe = topk_indices.clamp(min=0, max=logits.shape[-1] - 1).to(torch.int64)
    return torch.where(valid, torch.gather(logits, -1, safe), float("-inf"))


def wrapper(
    index_q: Tensor,
    index_k: Tensor,
    weights: Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: Tensor | None = None,
    cu_seqlen_ks: Tensor | None = None,
    cu_seqlen_ke: Tensor | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the Triton lighting indexer for compressed-KV selection, under the TileLang row's contract.

    ``index_q`` is ``[S, B, H, D]``, ``index_k`` is ``[S_kv, B, D]``, ``weights`` is ``[S, B, H]``,
    in any strides. Returns fp32 scores (without gradient) and int32 indices with shape
    ``[B, S, K]``; ``K`` is the supplied ``topk_indices`` width when present, otherwise
    ``min(topk, S_kv)``, and unselectable slots hold index ``-1`` and score ``-inf``.
    ``cu_seqlen_ks`` / ``cu_seqlen_ke`` already encode packed ranges, so
    ``packed_sequence_slices`` is unused.
    """
    del packed_sequence_slices
    if (cu_seqlen_ks is None) != (cu_seqlen_ke is None):
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be provided together")
    seq_len, batch, heads, dim = index_q.shape
    seq_len_kv = index_k.shape[0]
    if heads > _Q_ROWS or heads & (heads - 1):
        raise ValueError(f"The Triton DeepSeek-V4 indexer requires a power-of-two head count <= 128, got {heads}")
    if dim < 16 or dim & (dim - 1):
        raise ValueError(f"The Triton DeepSeek-V4 indexer requires a power-of-two head dim >= 16, got {dim}")
    if index_q.dtype not in (torch.bfloat16, torch.float16) or index_k.dtype != index_q.dtype:
        raise ValueError(
            f"The Triton DeepSeek-V4 indexer requires bfloat16 or float16 q/k of one dtype, "
            f"got q={index_q.dtype}, k={index_k.dtype}"
        )
    if cu_seqlen_ks is None:
        cu_seqlen_ks, cu_seqlen_ke = _causal_ranges(seq_len, compress_ratio, index_q.device)
    elif cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            "Packed indexer ranges must have shape "
            f"({seq_len},), got {tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
        )

    with torch.no_grad():
        logits = torch.stack(
            [
                indexer_logits(index_q[:, b], index_k[:, b], weights[:, b], cu_seqlen_ks, cu_seqlen_ke)
                for b in range(batch)
            ]
        )
        if topk_indices is None:
            top_k = min(topk, seq_len_kv)
            if top_k == 0:
                topk_indices = torch.empty((batch, seq_len, 0), device=logits.device, dtype=torch.int32)
            else:
                scores, topk_indices = logits.topk(top_k, dim=-1)
                topk_indices = topk_indices.to(torch.int32).masked_fill(scores == float("-inf"), -1)
        return _gather_scores(logits, topk_indices), topk_indices
