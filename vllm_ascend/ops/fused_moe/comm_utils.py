# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
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
#
import torch
import torch.distributed
import torch.distributed as dist
import torch_npu

COMM_STREAM = None


@torch.library.custom_op("vllm_ascend::fxrt_all_to_all_wait", mutates_args=())
def _fxrt_all_to_all_wait(
    input_: torch.Tensor,
    output_splits: torch.Tensor | None,
    input_splits: torch.Tensor | None,
    output_capacity: int,
    group_name: str,
) -> torch.Tensor:
    # Keep Work and its wait inside one opaque boundary. No Work object can
    # escape into Dynamo, and consumers see a completed communication.
    group = dist.distributed_c10d._resolve_process_group(group_name)
    output_split_sizes = None if output_splits is None else output_splits.to(torch.int64).cpu().tolist()
    input_split_sizes = None if input_splits is None else input_splits.to(torch.int64).cpu().tolist()
    actual_output_tokens = input_.shape[0] if output_split_sizes is None else sum(output_split_sizes)
    actual_input_tokens = input_.shape[0] if input_split_sizes is None else sum(input_split_sizes)
    actual_output = input_.new_empty((actual_output_tokens, *input_.shape[1:]))
    handle = dist.all_to_all_single(
        actual_output,
        input_[:actual_input_tokens].contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    handle.wait()
    output = input_.new_zeros((output_capacity, *input_.shape[1:]))
    output[:actual_output_tokens].copy_(actual_output)
    return output


@_fxrt_all_to_all_wait.register_fake
def _fxrt_all_to_all_wait_fake(input_, output_splits, input_splits, output_capacity, group_name):
    return input_.new_empty((output_capacity, *input_.shape[1:]))


class _CompletedAllToAll:
    def wait(self):
        return True


@torch.library.custom_op("vllm_ascend::fxrt_unpermute_nonempty", mutates_args=())
def fxrt_unpermute_nonempty(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] == 0:
        return tokens.clone()
    return torch_npu.npu_moe_token_unpermute(tokens, indices)


@fxrt_unpermute_nonempty.register_fake
def _fxrt_unpermute_nonempty_fake(tokens, indices):
    return torch.empty_like(tokens)


@torch.library.custom_op("vllm_ascend::fxrt_alltoall_preprocess", mutates_args=())
def fxrt_alltoall_preprocess(
    topk_ids: torch.Tensor,
    expert_ids_per_ep_rank: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    first_local_expert: int,
    ep_size: int,
    routing_capacity: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build AllToAllV routing metadata behind one opaque boundary.

    The implementation intentionally contains the routing-count collective
    and all value-dependent operations.  In particular, histogram reduction,
    D2H conversion by the subsequent collective, and repeat_interleave's
    data-dependent output size must not be visible in the FX graph.
    """
    num_local_tokens_per_expert = torch.histc(topk_ids, bins=num_experts, min=0, max=num_experts)
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)

    group = dist.distributed_c10d._resolve_process_group(group_name)
    gathered_counts = torch.empty(
        (ep_size * num_experts,),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(gathered_counts, num_local_tokens_per_expert.contiguous(), group=group)
    local_counts = gathered_counts.reshape(ep_size, num_experts)[
        :, first_local_expert : first_local_expert + num_local_experts
    ]
    output_splits = local_counts.sum(dim=-1)
    tokens_per_local_expert = local_counts.sum(dim=0)
    if num_local_experts > 1:
        global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, local_counts.ravel())
    else:
        global_indices = expert_ids_per_ep_rank.new_empty((0,))
    actual_tokens = global_indices.shape[0]
    padding = routing_capacity - actual_tokens
    if padding:
        # Assign zero-filled padding rows to the final local expert so grouped
        # matmul consumes the entire fixed-capacity buffer consistently.
        padding_indices = expert_ids_per_ep_rank.new_full((padding,), num_local_experts - 1)
        global_indices = torch.cat((global_indices, padding_indices))
        tokens_per_local_expert = tokens_per_local_expert.clone()
        tokens_per_local_expert[-1] += padding
    return (
        tokens_per_local_expert.to(torch.int64),
        input_splits.to(torch.int64),
        output_splits.to(torch.int64),
        global_indices,
    )


@fxrt_alltoall_preprocess.register_fake
def _fxrt_alltoall_preprocess_fake(
    topk_ids,
    expert_ids_per_ep_rank,
    num_experts,
    num_local_experts,
    first_local_expert,
    ep_size,
    routing_capacity,
    group_name,
):
    count_dtype = torch.int64
    tokens_per_local_expert = torch.empty(
        (num_local_experts,), dtype=count_dtype, device=topk_ids.device
    )
    input_splits = torch.empty((ep_size,), dtype=count_dtype, device=topk_ids.device)
    output_splits = torch.empty((ep_size,), dtype=count_dtype, device=topk_ids.device)
    global_indices = expert_ids_per_ep_rank.new_empty((routing_capacity,))
    return tokens_per_local_expert, input_splits, output_splits, global_indices


def async_all_to_all(
    input_,
    output_split_sizes,
    input_split_sizes,
    group,
    event=None,
    output_capacity=None,
):
    from vllm_ascend.utils import fxrt_prefill_decompose_enabled

    if torch.compiler.is_compiling() and fxrt_prefill_decompose_enabled() and event is None:
        if output_capacity is None:
            output_capacity = input_.shape[0] * torch.distributed.get_world_size(group)
        a2a_out = _fxrt_all_to_all_wait(
            input_,
            output_split_sizes,
            input_split_sizes,
            output_capacity,
            group.group_name,
        )
        return input_, a2a_out, _CompletedAllToAll()

    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)
