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

"""mHC pre AITER adapter (AMD ROCm)."""

from __future__ import annotations

from torch import Tensor

from ..helper import POST_MULTIPLIER, aiter_dsv4_mhc, require_aiter_input


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """AITER pre / Sinkhorn / collapse: Triton forward, PyTorch-recompute backward. Lazy-imports ``aiter``."""
    require_aiter_input(hidden_streams)
    batch, seq_len, hc, hidden = hidden_streams.shape
    post, comb, collapsed = aiter_dsv4_mhc().mhc_pre_dsv4(
        hidden_streams.reshape(batch * seq_len, hc, hidden).contiguous(),
        fn.float().contiguous(),
        scale.float().contiguous(),
        base.float().contiguous(),
        rms_eps=norm_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=POST_MULTIPLIER,
        sinkhorn_repeat=sinkhorn_iters,
    )
    return (
        post.view(batch, seq_len, hc),
        comb.view(batch, seq_len, hc, hc),
        collapsed.view(batch, seq_len, hidden),
    )
