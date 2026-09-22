# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""V4.1 CV prolog stages: keep stream/event objects out of the FX graph.

The three boundaries preserve Q-Cube/KV-Vector and Q-Vector/KV-Cube overlap.
Compressor, indexer, sparse attention and output projection remain traceable.
Weights and persistent events are owned by the registered attention layer,
resolved in the same way as vllm::dsa_v41_forward.
"""

import torch
from vllm.forward_context import get_forward_context


def _layer(name):
    return get_forward_context().no_compile_layers[name]


def _wrappers(attn):
    impl = attn.dsa_attn.dsa_attn.impl
    return impl.cv_wq_a, impl.cv_wkv, impl.cv_wq_b


def _aux_stream():
    from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream

    return dsv4_dsa_overlap_stream()


def _record_inputs(stream, *values):
    # Inputs allocated on the main stream can be released by FXRT immediately
    # after the call. The allocator must retain them until aux work completes.
    for value in values:
        if value is not None:
            value.record_stream(stream)


def _quant_meta(wrapper, x):
    if wrapper._is_w8a8_dynamic and not wrapper._has_communication:
        return torch.empty_like(x, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)
    # CVLinearWrapper.quantize returns x unchanged for FP8/BF16/communication.
    # None denotes that passthrough, avoiding aliases in custom-op outputs.
    return None, None


def _output_dtype(wrapper, input_dtype):
    if wrapper._is_w8a8_dynamic and not wrapper._has_communication:
        return wrapper.linear.weight_scale.dtype
    return input_dtype


@torch.library.custom_op("vllm_ascend::fxrt_dsv41_prolog_q_a", mutates_args=())
def prolog_q_a(x: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = _layer(layer_name)
    wq_a, wkv, _ = _wrappers(attn)
    main, aux = torch.npu.current_stream(), _aux_stream()
    q_quant, q_scale = wq_a.quantize(x)
    share_quant = type(wq_a._quant_method) is type(wkv._quant_method) and (
        wq_a._has_communication == wkv._has_communication
    )
    if share_quant:
        kv_quant, kv_scale = q_quant, q_scale
        ready = None
    else:
        start = main.record_event()
        with torch.npu.stream(aux):
            aux.wait_event(start)
            _record_inputs(aux, x)
            kv_quant, kv_scale = wkv.quantize(x)
            ready = aux.record_event()
        _record_inputs(main, kv_quant, kv_scale)
    q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)
    if ready is not None:
        main.wait_event(ready)
    if kv_quant is x:
        kv_quant = torch.empty_like(x)
        kv_scale = x.new_empty(0, dtype=torch.float32)
    return q_a, kv_quant, kv_scale


@prolog_q_a.register_fake
def _prolog_q_a_fake(x, layer_name):
    attn = _layer(layer_name)
    wq_a, wkv, _ = _wrappers(attn)
    share_quant = type(wq_a._quant_method) is type(wkv._quant_method) and (
        wq_a._has_communication == wkv._has_communication
    )
    kv_quant, kv_scale = _quant_meta(wq_a if share_quant else wkv, x)
    q_a = x.new_empty((*x.shape[:-1], attn.q_lora_rank), dtype=_output_dtype(wq_a, x.dtype))
    if kv_quant is None:
        kv_quant = torch.empty_like(x)
        kv_scale = x.new_empty(0, dtype=torch.float32)
    return q_a, kv_quant, kv_scale


@torch.library.custom_op("vllm_ascend::fxrt_dsv41_prolog_q_norm", mutates_args=())
def prolog_q_norm(
    x: torch.Tensor, q_a: torch.Tensor, kv_quant: torch.Tensor,
    kv_scale: torch.Tensor, layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = _layer(layer_name)
    _, wkv, wq_b = _wrappers(attn)
    main, aux = torch.npu.current_stream(), _aux_stream()
    start = main.record_event()
    with torch.npu.stream(aux):
        aux.wait_event(start)
        kv_input = x if kv_quant.numel() == 0 else kv_quant
        scale = None if kv_scale.numel() == 0 else kv_scale
        _record_inputs(aux, kv_input, scale)
        kv = wkv.matmul(kv_input, scale, bias=attn.wkv.bias)
        ready = aux.record_event()
    qr = attn.q_norm(q_a)
    q_b_quant, q_b_scale = wq_b.quantize(qr)
    main.wait_event(ready)
    _record_inputs(main, kv)
    if q_b_quant is qr:
        q_b_quant = torch.empty_like(qr)
        q_b_scale = qr.new_empty(0, dtype=torch.float32)
    return qr, q_b_quant, q_b_scale, kv


@prolog_q_norm.register_fake
def _prolog_q_norm_fake(x, q_a, kv_quant, kv_scale, layer_name):
    attn = _layer(layer_name)
    _, wkv, wq_b = _wrappers(attn)
    qr = torch.empty_like(q_a)
    q_b_quant, q_b_scale = _quant_meta(wq_b, qr)
    kv = x.new_empty((*x.shape[:-1], attn.head_dim), dtype=_output_dtype(wkv, x.dtype))
    return qr, q_b_quant, q_b_scale, kv


@torch.library.custom_op("vllm_ascend::fxrt_dsv41_prolog_q_b", mutates_args=("caches",))
def prolog_q_b(
    x: torch.Tensor, qr: torch.Tensor, q_b_quant: torch.Tensor,
    q_b_scale: torch.Tensor, kv: torch.Tensor, cos: torch.Tensor,
    sin: torch.Tensor, caches: list[torch.Tensor], layer_name: str,
) -> torch.Tensor:
    attn = _layer(layer_name)
    impl = attn.v41_impl
    _, _, wq_b = _wrappers(attn)
    metadata = impl._get_layer_metadata(get_forward_context().attn_metadata)
    main, aux = torch.npu.current_stream(), _aux_stream()
    start = main.record_event()
    with torch.npu.stream(aux):
        aux.wait_event(start)
        _record_inputs(aux, kv, cos, sin, *caches)
        normalized = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            normalized.unsqueeze(1), cos, sin, rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        impl._scatter_swa_kv(attn, normalized.squeeze(1), metadata.swa)
        ready = aux.record_event()
    q_input = qr if q_b_quant.numel() == 0 else q_b_quant
    q_scale = None if q_b_scale.numel() == 0 else q_b_scale
    q = wq_b.matmul(q_input, q_scale, bias=attn.wq_b.bias)
    q = impl._reshape_query_heads(q, attn.head_dim)
    main.wait_event(ready)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        q.unsqueeze(1), cos, sin, rotary_mode="interleave",
        partial_slice=[attn.nope_head_dim, attn.head_dim],
    )
    return q.to(x.dtype)


@prolog_q_b.register_fake
def _prolog_q_b_fake(x, qr, q_b_quant, q_b_scale, kv, cos, sin, caches, layer_name):
    attn = _layer(layer_name)
    return x.new_empty((x.shape[0], attn.n_local_heads, attn.head_dim))


@torch.library.custom_op("vllm_ascend::fxrt_dsv41_wait_metadata", mutates_args=())
def wait_metadata(stage: int, group_id: int) -> None:
    from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata

    wait_for_device_metadata(DeviceMetadataStage(stage), group_id)


@wait_metadata.register_fake
def _wait_metadata_fake(stage, group_id):
    return None
