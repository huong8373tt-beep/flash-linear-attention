# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.modules.activations.backends.triton_ascend import TritonAscendBackend as ActivationsBackend
from fla.modules.fused_cross_entropy.backends.triton_ascend import TritonAscendBackend as CrossEntropyBackend
from fla.modules.fused_kl_div.backends.triton_ascend import TritonAscendBackend as KLDivBackend
from fla.modules.fused_linear_cross_entropy.backends.triton_ascend import TritonAscendBackend as LinearCrossEntropyBackend
from fla.modules.grpo.backends.triton_ascend import TritonAscendBackend as GRPOBackend
from fla.modules.rotary.backends.triton_ascend import TritonAscendBackend as RotaryBackend


class TritonAscendBackend(
    ActivationsBackend,
    CrossEntropyBackend,
    KLDivBackend,
    LinearCrossEntropyBackend,
    GRPOBackend,
    RotaryBackend,
):
    """Legacy aggregate of the operation-owned Ascend adapters."""
