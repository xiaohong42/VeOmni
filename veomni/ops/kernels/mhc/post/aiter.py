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

"""mHC post AITER adapter (AMD ROCm)."""

from __future__ import annotations

from torch import Tensor

from ..helper import aiter_dsv4_mhc, require_aiter_input


def wrapper(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> Tensor:
    """Mix *output* back into the residual streams, as the eager row does. Lazy-imports ``aiter``.

    *post* is ``[B, S, H]`` and *comb* is ``[B, S, H, H]``. AITER runs a Triton forward and
    owns the PyTorch-recompute backward.
    """
    require_aiter_input(residual)
    batch, seq_len, hc, hidden = residual.shape
    mixed = aiter_dsv4_mhc().mhc_post_dsv4(
        output.reshape(batch * seq_len, hidden),
        residual.reshape(batch * seq_len, hc, hidden),
        post.float().reshape(batch * seq_len, hc),
        comb.float().reshape(batch * seq_len, hc, hc),
    )
    return mixed.view(batch, seq_len, hc, hidden)
