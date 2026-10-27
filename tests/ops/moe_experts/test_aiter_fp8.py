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

"""AITER blockwise FP8 routed experts (AMD ROCm) against a PyTorch oracle of the same quantization."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from tests.ops.moe_experts.reference import standard_ep_reference
from veomni.ops import resolve_op
from veomni.ops.platform import ROCM_GPU


requires_rocm_aiter = pytest.mark.skipif(
    not (ROCM_GPU.matches() and importlib.util.find_spec("aiter") is not None),
    reason="The AITER FP8 MoE backend targets AMD ROCm GPUs with aiter installed.",
)


def _fp8():
    from veomni.ops.kernels.moe_experts.shared.aiter_fp8 import fp8_dtype

    return fp8_dtype()


def _round_trip(blocks, dims):
    dtype = _fp8()
    # A tensor divisor: PyTorch multiplies by a rounded reciprocal when dividing by a Python scalar.
    fp8_max = torch.tensor(torch.finfo(dtype).max, device=blocks.device)
    scales = blocks.abs().amax(dims, keepdim=True).clamp_min(1e-4) / fp8_max
    return (blocks / scales).to(dtype).float() * scales


def _quantized_rows(x):
    return _round_trip(x.float().reshape(x.shape[0], -1, 128), -1).reshape(x.shape)


def _quantized_columns(x):
    padded = F.pad(x.float(), (0, 0, 0, (-x.shape[0]) % 128))
    return _round_trip(padded.reshape(-1, 128, x.shape[1]), 1).reshape(padded.shape)[: x.shape[0]]


def _quantized_weight(w):
    n, k = w.shape
    return _round_trip(w.float().reshape(n // 128, 128, k // 128, 128), (1, 3)).reshape(n, k)


def _relative_error(actual, expected, limit, label):
    assert torch.isfinite(actual).all(), label
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert error < limit, f"{label}: relative L2 {error.item():.6f} >= {limit}"


class _ReferenceLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return (_quantized_rows(x) @ _quantized_weight(w).T).bfloat16()

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        dx = (_quantized_rows(g) @ _quantized_weight(w)).bfloat16()
        dw = (_quantized_columns(g).T @ _quantized_columns(x)).bfloat16()
        return dx, dw


def _quantized_moe_reference(x, routing, selected, w1, w2, limit):
    output = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(selected == expert)
        if rows.numel() == 0:
            continue
        gate, up = _ReferenceLinear.apply(x[rows], w1[expert]).chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        values = _ReferenceLinear.apply(F.silu(gate) * up, w2[expert])
        output.index_add_(0, rows, (values * routing[rows, slots, None]).to(x.dtype))
    return output


@requires_rocm_aiter
def test_quantizers_match_the_reference_round_trip():
    """Rows, per-expert token-axis columns and weight tiles dequantize to the oracle's exact values."""
    from veomni.ops.kernels.moe_experts.shared.aiter_fp8 import (
        GroupedLayout,
        quantize_columns,
        quantize_rows,
        quantize_weights,
    )

    torch.manual_seed(3)
    counts = torch.tensor([130, 0, 1, 257])
    x = torch.randn(int(counts.sum()), 384, device="cuda", dtype=torch.bfloat16) * 3
    layout = GroupedLayout.from_counts(counts, x.device)

    q, s = quantize_rows(x)
    assert q.dtype == _fp8() and s.shape == (x.shape[0], 3)
    restored = q.float() * s.repeat_interleave(128, dim=1)
    torch.testing.assert_close(restored, _quantized_rows(x), rtol=0, atol=0)

    q, s = quantize_columns(x, layout)
    # Scale blocks restart at every expert: 130 -> 2, 0 -> 0, 1 -> 1, 257 -> 3.
    assert s.shape == (2 + 1 + 3, 384)
    restored = q.float() * s.repeat_interleave(layout.blocks[:, 1].long(), dim=0)
    expected = torch.cat([_quantized_columns(part) for part in x.split(counts.tolist()) if part.numel()])
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    w = torch.randn(3, 256, 384, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_weights(w)
    assert s.shape == (3, 2, 3)
    restored = q.float() * s.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    torch.testing.assert_close(restored, torch.stack([_quantized_weight(t) for t in w]), rtol=0, atol=0)


@requires_rocm_aiter
@pytest.mark.parametrize("counts", [(200, 0, 77), (1, 128, 129)])
def test_grouped_linear_matches_the_reference_per_expert(counts):
    from veomni.ops.kernels.moe_experts.shared.aiter_fp8 import GroupedLayout, GroupedLinear

    torch.manual_seed(5)
    total, n, k = sum(counts), 256, 384
    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(len(counts), n, k, device="cuda") * 0.05).bfloat16().requires_grad_()
    grad = torch.randn(total, n, device="cuda", dtype=torch.bfloat16)
    layout = GroupedLayout.from_counts(torch.tensor(counts), x.device)

    y = GroupedLinear.apply(x, w, layout)
    dx, dw = torch.autograd.grad(y, (x, w), grad)

    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    yr = torch.cat(
        [_ReferenceLinear.apply(part, wr[e]) for e, part in enumerate(xr.split(list(counts))) if part.numel()]
    )
    dxr, dwr = torch.autograd.grad(yr, (xr, wr), grad)
    _relative_error(y, yr, 1e-2, "output")
    _relative_error(dx, dxr, 1e-2, "input gradient")
    for e, count in enumerate(counts):
        if count:
            _relative_error(dw[e], dwr[e], 1e-2, f"weight gradient of expert {e}")
        else:
            assert torch.count_nonzero(dw[e]) == 0


@requires_rocm_aiter
@pytest.mark.parametrize("merged,limit,routing_dtype", [(True, None, torch.float32), (False, 0.7, torch.bfloat16)])
def test_full_moe_forward_backward(merged, limit, routing_dtype):
    torch.manual_seed(19)
    e, t, h, intermediate = 4, 139, 256, 128
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w1 = (torch.randn(e, 2 * intermediate, h, device="cuda") * 0.03).bfloat16()
    w2 = (torch.randn(e, h, intermediate, device="cuda") * 0.03).bfloat16()
    routing = torch.rand(t, 2, device="cuda", dtype=routing_dtype)
    # Expert 3 is empty. Duplicate assignments are legal on the non-EP path.
    selected = torch.randint(0, e - 1, (t, 2), device="cuda")
    grad = torch.randn_like(x) * 0.1
    backend = resolve_op("moe_experts", "standard", "aiter_fp8").wrapper
    empty = torch.empty(0, device="cuda")
    results = []
    for reference in ("kernel", "bf16", "quantized"):
        leaves = [a.detach().clone().requires_grad_() for a in (x, routing, w1, w2)]
        xi, ri, wi, wo = leaves
        a, b = wi.chunk(2, dim=1)
        if reference == "bf16":
            y = standard_ep_reference(xi, ri, selected, a, b, wo, num_experts=e, swiglu_limit=limit)
        elif reference == "quantized":
            y = _quantized_moe_reference(xi, ri, selected, wi, wo, limit)
        else:
            y = backend(
                xi,
                ri,
                selected,
                empty if merged else a,
                empty if merged else b,
                wo,
                wi if merged else empty,
                num_experts=e,
                swiglu_limit=limit,
            )
        results.append((y, *torch.autograd.grad(y, leaves, grad)))
    for label, actual, bf16, quantized in zip(("output", "hidden", "router", "gate_up", "down"), *results):
        _relative_error(actual, quantized, 0.01, f"quantized {label}")
        # The derivative at the clamp boundary is discontinuous; use the
        # independent FP8 oracle above for that deliberately adversarial case.
        if limit is None or label not in ("hidden", "gate_up"):
            # e4m3fnuz keeps e4m3fn's 3 mantissa bits, so the FP8 error budget against BF16 is the same.
            _relative_error(actual, bf16, 0.085, f"BF16 {label}")
    assert torch.count_nonzero(results[0][-1][-1]) == 0
    assert torch.count_nonzero(results[0][-2][-1]) == 0


@requires_rocm_aiter
def test_empty_moe_has_zero_gradients():
    backend = resolve_op("moe_experts", "standard", "aiter_fp8").wrapper
    x = torch.randn(0, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(2, 256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    routing = torch.rand(0, 1, device="cuda", requires_grad=True)
    selected = torch.zeros(0, 1, device="cuda", dtype=torch.int64)
    empty = torch.empty(0, device="cuda")
    y = backend(x, routing, selected, empty, empty, w2, w1, num_experts=2)
    assert y.shape == x.shape
    grads = torch.autograd.grad(y.sum(), (w1, w2), allow_unused=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)


@pytest.mark.parametrize(
    ("bad", "match"),
    (
        ("device", "AMD ROCm GPU"),
        ("hidden", "multiples of 128"),
    ),
)
def test_invalid_operands_fail_explicitly(bad, match):
    # The module's wrapper itself, since the registry only resolves the row on a ROCm GPU.
    from veomni.ops.kernels.moe_experts.standard.aiter import wrapper as backend

    on_rocm = ROCM_GPU.matches()
    if bad == "hidden" and not on_rocm:
        pytest.skip("shape checks run after the device check, which needs a ROCm GPU")
    device = "cuda" if (on_rocm and bad != "device") else "cpu"
    h = 200 if bad == "hidden" else 256
    x = torch.randn(4, h, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(2, 256, h, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(2, h, 128, device=device, dtype=torch.bfloat16)
    routing = torch.rand(4, 1, device=device)
    selected = torch.zeros(4, 1, device=device, dtype=torch.int64)
    empty = torch.empty(0, device=device)
    with pytest.raises((RuntimeError, ValueError), match=match):
        backend(x, routing, selected, empty, empty, w2, w1, num_experts=2)
