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

"""Blockwise FP8 routed experts on AMD ROCm through AITER's grouped GEMM (``aiter_fp8``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state


def _swiglu(projected: Tensor, limit: float | None) -> Tensor:
    gate, up = projected.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _experts(x, counts, fc1, fc2, swiglu_limit):
    from ..shared.aiter_fp8 import GroupedLayout, GroupedLinear

    layout = GroupedLayout.from_counts(counts, x.device)
    projected = GroupedLinear.apply(x, fc1, layout)
    return GroupedLinear.apply(_swiglu(projected, swiglu_limit), fc2, layout)


def _ep_forward(x, routing_weights, selected_experts, fc1, fc2, num_experts, swiglu_limit):
    """Dispatch tokens across EP ranks, run FP8 local experts, and combine outputs."""
    from .....distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all

    ep_group = get_parallel_state().ep_group
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    input_splits, output_splits, per_rank_counts, counts = preprocess(expert_mask, num_experts, ep_group)
    tokens, routing_map, permutation, original_shape = token_pre_all2all(
        x,
        expert_mask,
        num_experts,
        input_splits,
        output_splits,
        per_rank_counts,
        ep_group,
    )
    del expert_mask
    outputs = _experts(tokens, counts, fc1, fc2, swiglu_limit)
    del tokens
    return tokens_post_all2all(
        outputs,
        routing_weights,
        selected_experts,
        num_experts,
        input_splits,
        output_splits,
        per_rank_counts,
        routing_map,
        permutation,
        original_shape,
        ep_group,
    )


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_weight: Tensor,
    fc1_2_weight: Tensor,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
    assume_distinct_experts: bool = False,
) -> Tensor:
    """Run FP8 expert GEMMs with BF16 inputs, outputs and first-order gradients on AMD ROCm.

    The contract is ``deepgemm_fp8``'s: merged or split gate/up weights,
    zero-token experts, and EP with distinct or repeated expert ids; hidden and
    intermediate sizes must be multiples of 128; routing weights are applied
    after the down projection. Unsupported hardware, dtypes or layouts fail
    explicitly. Expert counts are read on the host to build the layout.
    """
    if not hidden_states.is_cuda or torch.version.hip is None:
        raise RuntimeError("aiter_fp8 requires an AMD ROCm GPU.")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("aiter_fp8 requires BF16 hidden_states with shape [tokens, hidden].")
    if (
        selected_experts.ndim != 2
        or selected_experts.shape[1] == 0
        or routing_weights.shape != selected_experts.shape
        or selected_experts.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Routing weights and expert indices must have matching [tokens, top_k] shapes.")
    if selected_experts.dtype != torch.int64 or not routing_weights.is_floating_point():
        raise ValueError("Expert indices must be int64 and routing weights must be floating point.")
    if selected_experts.device != hidden_states.device or routing_weights.device != hidden_states.device:
        raise ValueError("Routing tensors must be on the same device as hidden_states.")
    merged = fc1_1_2_weight.numel() != 0
    split = fc1_1_weight.numel() != 0 or fc1_2_weight.numel() != 0
    if merged == split or (split and (fc1_1_weight.numel() == 0 or fc1_2_weight.numel() == 0)):
        raise ValueError("Provide either both split fc1 weights or merged fc1 weights, not both or neither.")
    fc1 = fc1_1_2_weight if merged else torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
    if fc1.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError("Expert weights must have shapes [E,2*I,H] and [E,H,I].")
    e, h, intermediate = fc2_weight.shape
    if fc1.shape != (e, 2 * intermediate, h) or hidden_states.shape[1] != h:
        raise ValueError("Incompatible expert weights or hidden dimension.")
    if min(e, h, intermediate) == 0 or h % 128 or intermediate % 128:
        raise ValueError("FP8 MoE requires positive dimensions and hidden/intermediate multiples of 128.")
    for weight in (fc1, fc2_weight):
        if weight.dtype != torch.bfloat16 or weight.device != hidden_states.device:
            raise ValueError("FP8 MoE weights must be BF16 and on the same device as hidden_states.")
    fc1, fc2 = fc1.contiguous(), fc2_weight.contiguous()
    if get_parallel_state().ep_enabled:
        if e * get_parallel_state().ep_size != num_experts:
            raise ValueError("The EP-local expert weight count does not match num_experts / ep_size.")
        return _ep_forward(hidden_states, routing_weights, selected_experts, fc1, fc2, num_experts, swiglu_limit)
    if e != num_experts:
        raise ValueError("Expert weights must contain num_experts experts when EP is disabled.")

    # Each top-k slot remains a separate row, including duplicate expert ids.
    order = torch.argsort(selected_experts.reshape(-1), stable=True)
    token_indices = torch.div(order, selected_experts.shape[1], rounding_mode="floor")
    counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
    if counts.numel() != num_experts:
        raise ValueError("Expert indices must be less than num_experts.")
    x = hidden_states.index_select(0, token_indices)
    y = _experts(x, counts, fc1, fc2, swiglu_limit)
    weights = routing_weights.reshape(-1).index_select(0, order)
    weighted = (y * weights[:, None]).to(hidden_states.dtype)
    return torch.zeros_like(hidden_states).index_add(0, token_indices, weighted)
