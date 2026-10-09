# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compatibility entry point for aggregate module dispatch."""

import warnings

from fla.backends import _registry_for, dispatch
from fla.modules.backends.triton_ascend import TritonAscendBackend

warnings.warn(
    "fla.modules.backends is deprecated and will be removed in the next release after 0.6.0. "
    "Import dispatch from fla.backends and use an operation-specific key instead.",
    DeprecationWarning,
    stacklevel=2,
)

modules_registry = _registry_for("modules")

modules_registry.register(TritonAscendBackend())
__all__ = ['dispatch', 'modules_registry']
