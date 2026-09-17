# SPDX-License-Identifier: Apache-2.0
"""Runtime stream boundaries for decomposed DSV4 shared experts and gate.

Stream/Event Python objects never cross the FX graph boundary. Layer lookup
uses the same forward-context registry as vllm::moe_forward_shared. Explicit
weight inputs keep their lifetimes and dependencies visible to the backend.
"""

import torch


def _layer(name):
    from vllm.forward_context import get_forward_context

    return get_forward_context().no_compile_layers[name]


@torch.library.custom_op("vllm_ascend::fxrt_moe_gate_overlap", mutates_args=())
def gate_overlap(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    weights: list[torch.Tensor],
    layer_name: str,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = _layer(layer_name)
    stream = layer.gate_stream
    assert stream is not None
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        return layer._gate_overlap_body(hidden_states, router_logits)


@gate_overlap.register_fake
def _gate_overlap_fake(hidden_states, router_logits, weights, layer_name, top_k):
    return (
        torch.empty_like(hidden_states),
        router_logits.new_empty((hidden_states.shape[0], top_k)),
        hidden_states.new_empty((hidden_states.shape[0], top_k), dtype=torch.int32),
    )


@torch.library.custom_op("vllm_ascend::fxrt_moe_overlap_wait", mutates_args=())
def overlap_wait(retained_inputs: list[torch.Tensor], stream_index: int) -> None:
    # Keep graph-owned buffers live until their side-stream consumers finish.
    # A scalar-only wait expresses ordering but not allocator dependencies.
    from vllm_ascend.ops.fxrt_side_effects import _stream_from_index

    torch.npu.current_stream().wait_stream(_stream_from_index(stream_index))


@overlap_wait.register_fake
def _overlap_wait_fake(retained_inputs, stream_index):
    return None


@torch.library.custom_op("vllm_ascend::fxrt_moe_shared_overlap", mutates_args=())
def shared_overlap(
    hidden_states: torch.Tensor,
    routed_dependency: torch.Tensor,
    weights: list[torch.Tensor],
    layer_name: str,
    event_indices: list[int],
    swiglu_limit: float,
) -> torch.Tensor:
    from vllm_ascend.ops.fused_moe.fused_moe import FusedMoEEvents

    before, after, dispatch, gmm2, combine = (None if index < 0 else index for index in event_indices)
    return _layer(layer_name)._forward_shared_experts(
        hidden_states,
        FusedMoEEvents(
            before_routed_experts=before,
            after_routed_experts=after,
            before_dispatch=dispatch,
            before_gmm2=gmm2,
            before_combine=combine,
            swiglu_limit=swiglu_limit,
        ),
    )


@shared_overlap.register_fake
def _shared_overlap_fake(hidden_states, routed_dependency, weights, layer_name, event_indices, swiglu_limit):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("vllm_ascend::fxrt_moe_overlap_gather", mutates_args=())
def overlap_gather(x: torch.Tensor, output_rows: int, stream_index: int) -> torch.Tensor:
    if stream_index < 0:
        result = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(x, True, True)
        if result.shape[0] != output_rows:
            raise RuntimeError(f"MoE overlap gather rows mismatch: actual={result.shape[0]}, scheduler={output_rows}")
        # custom_op outputs must not alias inputs, including single-rank cases.
        return result.clone() if result is x else result
    from vllm_ascend.distributed.utils import fc3_all_gather_and_maybe_unpad_impl
    from vllm_ascend.ops.fxrt_side_effects import _stream_from_index

    stream = _stream_from_index(stream_index)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        result = fc3_all_gather_and_maybe_unpad_impl(x)
        if result.shape[0] != output_rows:
            raise RuntimeError(
                f"MoE overlap quant gather rows mismatch: actual={result.shape[0]}, scheduler={output_rows}"
            )
        return result


@overlap_gather.register_fake
def _overlap_gather_fake(x, output_rows, stream_index):
    return x.new_empty((output_rows, *x.shape[1:]))


@torch.library.custom_op("vllm_ascend::fxrt_moe_overlap_reduce", mutates_args=())
def overlap_reduce(x: torch.Tensor, output_rows: int) -> torch.Tensor:
    result = torch.ops.vllm.maybe_pad_and_reduce(x, True)
    if result.shape[0] != output_rows:
        raise RuntimeError(f"MoE overlap reduce rows mismatch: actual={result.shape[0]}, expected={output_rows}")
    return result.clone() if result is x else result


@overlap_reduce.register_fake
def _overlap_reduce_fake(x, output_rows):
    return x.new_empty((output_rows, *x.shape[1:]))
