# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Configuration and per-call isolation for V4.1 prefill DSA tracing."""

from vllm_ascend import envs


def dsv41_prefill_dsa_enabled(vllm_config) -> bool:
    from vllm_ascend.platform import _is_fxrt_backend_enabled

    kv = vllm_config.kv_transfer_config
    return (
        envs.VLLM_ASCEND_FXRT_DECOMPOSE_DSV41_PREFILL_DSA
        and _is_fxrt_backend_enabled(vllm_config)
        and not vllm_config.model_config.enforce_eager
        and (kv is None or (kv.is_kv_producer and not kv.is_kv_consumer))
    )


def is_dsv41_prefill_step(metadata) -> bool:
    # Warmup without attention metadata, decode and mixed P/D batches keep
    # the opaque operator. Token length alone cannot identify prefill.
    if metadata is None:
        return False
    return getattr(metadata, "num_prefills", 0) > 0 and getattr(
        metadata, "num_decode_tokens", 0
    ) == 0
