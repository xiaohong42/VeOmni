# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""First-party mHC math and TileKernels / AITER input validation helpers.

Eager inlines unweighted RMSNorm. TileLang adapters validate CUDA placement,
BF16 dtype, rank-4 layout, and ``hc_mult=4`` here. Registry requirements
separately restrict those rows to NVIDIA SM90 or later and AMD gfx942/gfx950
and require the ``tile_kernels`` package. AITER adapters validate placement,
dtype and layout here; their rows need AMD ROCm and the ``aiter`` package.
"""

from __future__ import annotations

from types import ModuleType

import torch
from torch import Tensor


POST_MULTIPLIER = 2.0


def unweighted_rms(x: Tensor, eps: float) -> Tensor:
    """RMSNorm without an affine weight. Matches ``DeepseekV4UnweightedRMSNorm``."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def require_tilelang_input(x: Tensor, hc_mult: int) -> None:
    """Validate tensor placement, dtype, shape, and hyper-connection width."""
    if not x.is_cuda:
        raise ValueError("TileKernels mHC requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC requires bfloat16 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"TileKernels mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
    if hc_mult != 4 or x.shape[-2] != 4:
        raise ValueError(f"TileKernels mHC backward currently requires hc_mult=4, got {hc_mult}")


def aiter_dsv4_mhc() -> ModuleType:
    """Import AITER's DeepSeek-V4 mHC ops, naming the missing piece on older AITER builds."""
    try:
        from aiter.ops.triton import fusions
    except ImportError as exc:
        raise ImportError("mhc_implementation='aiter' requires AITER's Triton mHC ops (aiter.ops.triton)") from exc
    if not hasattr(fusions, "mhc_pre_dsv4"):
        raise ImportError(
            "mhc_implementation='aiter' requires an AITER build with the DeepSeek-V4 mHC ops "
            "(aiter.ops.triton.fusions.mhc_pre_dsv4 / mhc_post_dsv4 / mhc_head_dsv4)"
        )
    return fusions


def require_aiter_input(x: Tensor) -> None:
    """Validate the placement, dtype and ``[batch, sequence, hc, hidden]`` layout AITER's mHC ops take."""
    if not x.is_cuda:
        raise ValueError("AITER mHC requires a GPU tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"AITER mHC requires bfloat16 or float16 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"AITER mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
