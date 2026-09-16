# SPDX-License-Identifier: Apache-2.0
"""Temporary Meta correction for the v0.23.0 RMS quantization kernel."""

import torch


def rms_norm_dynamic_quant_meta(x, gamma, smooth_scale=None, beta=None, epsilon=1e-6):
    # Keep symbolic dimensions and match the actual INT8 NPU output.
    return torch.empty_like(x, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)


# Retain the registration for the lifetime of each worker. Import only after
# the extension has registered its schema and before Dynamo traces the model.
_META_LIB = torch.library.Library("_C_ascend", "IMPL", "Meta")
if hasattr(torch.ops._C_ascend, "npu_rms_norm_dynamic_quant"):
    _META_LIB.impl("npu_rms_norm_dynamic_quant", rms_norm_dynamic_quant_meta, allow_override=True)
