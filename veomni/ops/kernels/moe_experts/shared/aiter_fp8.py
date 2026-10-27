# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""AMD ROCm FP8 preparation for AITER grouped linear training.

The recipe is ``deepgemm_fp8``'s: activations use 1x128 scales for forward and
dgrad and independent token-axis 128x1 scales for wgrad, weights use 128x128
scales, every scale is FP32, and every payload is quantized from the BF16
source. The payload is the GPU's native FP8 (e4m3fnuz on gfx942, e4m3fn
elsewhere), scaled to that format's largest finite value. AITER's
``grouped_gemm`` runs the three GEMMs on expert-contiguous rows without
padding, so token-axis scale blocks restart at every expert's first row.
This module is imported lazily by the ROCm-only backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .....utils.device import get_rocm_gcn_arch


_BLOCK = 128
_FNUZ_ARCHS = ("gfx940", "gfx941", "gfx942")


def fp8_dtype() -> torch.dtype:
    """Return the FP8 format AITER's Triton GEMMs consume natively on this GPU."""
    return torch.float8_e4m3fnuz if get_rocm_gcn_arch() in _FNUZ_ARCHS else torch.float8_e4m3fn


@dataclass
class GroupedLayout:
    """Expert-contiguous row offsets and the 128-row token-axis scale blocks of every expert.

    ``from_counts`` reads the counts on the host once; the layout is reused by
    both expert projections and all their backward GEMMs.
    """

    counts: tuple[int, ...]
    cu_seqlens: torch.Tensor
    blocks: torch.Tensor

    @classmethod
    def from_counts(cls, counts: torch.Tensor, device: torch.device) -> GroupedLayout:
        """Create the layout of ``counts`` rows per expert, in expert order."""
        logical = tuple(int(n) for n in counts.tolist())
        if any(n < 0 for n in logical):
            raise ValueError("Expert token counts must be nonnegative.")
        offsets, blocks = [0], []
        for n in logical:
            blocks.extend([offsets[-1] + start, min(_BLOCK, n - start)] for start in range(0, n, _BLOCK))
            offsets.append(offsets[-1] + n)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
        block_table = torch.tensor(blocks, dtype=torch.int32, device=device).reshape(-1, 2)
        return cls(logical, cu_seqlens, block_table)

    @property
    def rows(self) -> int:
        """Total routed rows."""
        return sum(self.counts)


# IEEE-rounded divisions: the scale is exactly amax / FP8_MAX and the payload x / scale.
@triton.jit
def _quant_scale(amax, FP8_MAX: tl.constexpr):
    return tl.fdiv(tl.maximum(amax, 1.0e-4), FP8_MAX, ieee_rounding=True)


@triton.jit
def _row_quant_kernel(X, Q, S, M, SX0, SX1, H: tl.constexpr, FP8_MAX: tl.constexpr, BM: tl.constexpr):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    x = tl.load(X + rows[:, None] * SX0 + cols[None, :] * SX1, mask=mask, other=0).to(tl.float32)
    scale = _quant_scale(tl.max(tl.abs(x), 1), FP8_MAX)
    tl.store(
        Q + rows[:, None] * H + cols[None, :],
        tl.fdiv(x, scale[:, None], ieee_rounding=True).to(Q.dtype.element_ty),
        mask=mask,
    )
    tl.store(S + rows * (H // 128) + tl.program_id(1), scale, mask=rows < M)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, H]`` BF16 rows in 1x128 blocks: FP8 ``[M, H]`` and FP32 scales ``[M, H / 128]``."""
    m, h = x.shape
    dtype = fp8_dtype()
    q = torch.empty((m, h), dtype=dtype, device=x.device)
    scales = torch.empty((m, h // _BLOCK), dtype=torch.float32, device=x.device)
    if m:
        _row_quant_kernel[(triton.cdiv(m, 16), h // _BLOCK)](
            x, q, scales, m, x.stride(0), x.stride(1), h, torch.finfo(dtype).max, 16
        )
    return q, scales


@triton.jit
def _column_quant_kernel(X, Q, S, Blocks, SX0, SX1, H: tl.constexpr, FP8_MAX: tl.constexpr, BN: tl.constexpr):
    block = tl.program_id(0)
    start = tl.load(Blocks + block * 2).to(tl.int64)
    count = tl.load(Blocks + block * 2 + 1)
    r = tl.arange(0, 128)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (r[:, None] < count) & (c[None, :] < H)
    x = tl.load(X + (start + r[:, None]) * SX0 + c[None, :] * SX1, mask=mask, other=0).to(tl.float32)
    scale = _quant_scale(tl.max(tl.abs(x), 0), FP8_MAX)
    tl.store(
        Q + (start + r[:, None]) * H + c[None, :],
        tl.fdiv(x, scale[None, :], ieee_rounding=True).to(Q.dtype.element_ty),
        mask=mask,
    )
    tl.store(S + block.to(tl.int64) * H + c, scale, mask=c < H)


def quantize_columns(x: torch.Tensor, layout: GroupedLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, H]`` BF16 rows along tokens, 128 rows of one expert per scale.

    Returns FP8 ``[M, H]`` in the input's row order and FP32 scales
    ``[sum_e ceil(rows_e / 128), H]``, the layout of AITER's grouped wgrad.
    """
    m, h = x.shape
    dtype = fp8_dtype()
    q = torch.empty((m, h), dtype=dtype, device=x.device)
    scales = torch.empty((layout.blocks.shape[0], h), dtype=torch.float32, device=x.device)
    if layout.blocks.shape[0]:
        _column_quant_kernel[(layout.blocks.shape[0], triton.cdiv(h, 32))](
            x, q, scales, layout.blocks, x.stride(0), x.stride(1), h, torch.finfo(dtype).max, 32
        )
    return q, scales


@triton.jit
def _weight_quant_kernel(W, Q, S, N: tl.constexpr, K: tl.constexpr, FP8_MAX: tl.constexpr):
    mb, kb, expert = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mb * 128 + tl.arange(0, 128)
    k = kb * 128 + tl.arange(0, 128)
    if N * K > 2147483648:
        m = m.to(tl.int64)
        k = k.to(tl.int64)
    offsets = expert.to(tl.int64) * N * K + m[:, None] * K + k[None, :]
    w = tl.load(W + offsets).to(tl.float32)
    scale = _quant_scale(tl.max(tl.max(tl.abs(w), 1), 0), FP8_MAX)
    tl.store(Q + offsets, tl.fdiv(w, scale, ieee_rounding=True).to(Q.dtype.element_ty))
    tl.store(S + (expert * (N // 128) + mb) * (K // 128) + kb, scale)


def quantize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[E, N, K]`` BF16 weights in 128x128 tiles: FP8 ``[E, N, K]`` and scales ``[E, N/128, K/128]``."""
    e, n, k = w.shape
    dtype = fp8_dtype()
    q = torch.empty((e, n, k), dtype=dtype, device=w.device)
    scales = torch.empty((e, n // _BLOCK, k // _BLOCK), dtype=torch.float32, device=w.device)
    if e:
        _weight_quant_kernel[(n // _BLOCK, k // _BLOCK, e)](w, q, scales, n, k, torch.finfo(dtype).max, num_warps=8)
    return q, scales


class GroupedLinear(torch.autograd.Function):
    """BF16-facing grouped linear with AITER FP8 GEMMs in forward, dgrad and wgrad.

    Rows of ``x`` are grouped by expert as in ``layout``; ``weight`` is
    ``[E, N, K]``. Quantization uses a straight-through training approximation.
    The weight is requantized for dgrad rather than keeping an FP8 copy across
    forward, and the wgrad operands are quantized from BF16 along tokens. AITER
    accumulates every GEMM in FP32. Higher-order gradients are unsupported.
    """

    @staticmethod
    def forward(ctx, x, weight, layout):
        """Return the BF16 ``[M, N]`` output and save the token-axis quantized input."""
        from aiter.ops.triton.moe.sonicmoe import grouped_gemm

        ctx.layout = layout
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        if x.shape[0]:
            qw, sw = quantize_weights(weight)
            qx, sx = quantize_rows(x)
            output = grouped_gemm(
                qx, qw, layout.cu_seqlens, B_is_transposed=True, A_scale=sx, B_scale=sw, out_dtype=x.dtype
            )
            del qw, sw, qx, sx
        else:
            output = x.new_empty((0, weight.shape[1]))
        qxc, sxc = quantize_columns(x, layout) if ctx.needs_input_grad[1] else (None, None)
        ctx.save_for_backward(qxc, sxc, weight if ctx.needs_input_grad[0] else None)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Compute input and weight gradients from independently quantized operands."""
        from aiter.ops.triton.moe.sonicmoe import grouped_gemm

        qxc, sxc, weight = ctx.saved_tensors
        layout = ctx.layout
        grad_output = grad_output.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            if grad_output.shape[0]:
                qw, sw = quantize_weights(weight)
                qg, sg = quantize_rows(grad_output)
                dx = grouped_gemm(qg, qw, layout.cu_seqlens, A_scale=sg, B_scale=sw, out_dtype=ctx.input_dtype)
                del qw, sw, qg, sg
            else:
                dx = grad_output.new_empty((0, weight.shape[2]), dtype=ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            qgc, sgc = quantize_columns(grad_output, layout)
            # The grouped wgrad writes every expert, including those without tokens (zeros).
            dw = grouped_gemm(
                qgc, qxc, layout.cu_seqlens, A_is_transposed=True, A_scale=sgc, B_scale=sxc, out_dtype=ctx.weight_dtype
            )
        return dx, dw, None
