# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
import torch_npu
import custom_ops
import cann_ops_transformer
from cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla_metadata,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41A5DraftSWASpec,
    DeepseekV41CompressorStateSpec,
    DeepseekV41DraftSWASpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
    is_v41_draft_swa_spec,
)
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend.utils import npu_stream_switch

_V41_PREFILL_WORKSPACES: dict = {}
_V41_PREFILL_WORKSPACE_BLOCKS_CACHE: dict = {}


def _prefill_workspace_blocks(vllm_config: VllmConfig, block_size: int, window_size: int) -> int:
    key = (id(vllm_config), block_size)
    entry = _V41_PREFILL_WORKSPACE_BLOCKS_CACHE.get(key)
    if entry is not None:
        return entry[1]
    from vllm_ascend.core.private_circle_pool import compute_private_circle_prefill_workspace_blocks

    blocks = compute_private_circle_prefill_workspace_blocks(
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        window_size=window_size,
        block_size=block_size,
    )
    # Strong ref to the config keeps the id() key unique.
    _V41_PREFILL_WORKSPACE_BLOCKS_CACHE[key] = (vllm_config, blocks)
    return blocks


def _get_private_circle_workspace(
    vllm_config: VllmConfig, dtype: torch.dtype, device: torch.device, layer_shape: torch.Size,
    num_blocks: int,
) -> torch.Tensor:
    key = (id(vllm_config), str(device), dtype, tuple(layer_shape))
    entry = _V41_PREFILL_WORKSPACES.get(key)
    workspace = entry[1] if entry is not None else None
    expected_shape = (num_blocks, *layer_shape)
    if workspace is None:
        workspace = torch.empty(expected_shape, dtype=dtype, device=device)
        _V41_PREFILL_WORKSPACES[key] = (vllm_config, workspace)
    elif tuple(workspace.shape) != expected_shape:
        raise RuntimeError(
            "V4.1 prefill workspace layout changed after initialization: "
            f"expected={expected_shape}, actual={tuple(workspace.shape)}"
        )
    return workspace


def _format_private_circle_slots(flat: torch.Tensor, block_size: int) -> torch.Tensor:
    slots = torch.full((flat.numel(), 2), -1, dtype=torch.int32, device=flat.device)
    valid = flat >= 0
    slots[:, 0].copy_(torch.where(valid, torch.div(flat, block_size, rounding_mode="floor"), -1))
    slots[:, 1].copy_(torch.where(valid, flat.remainder(block_size), -1))
    return slots


def _private_circle_batch_rows(
    common: Any, num_reqs: int, flags: torch.Tensor
) -> list[str]:
    """Describe every batch row for the decode-first violation log."""
    req_ids = getattr(common, "private_circle_req_ids", None) or []
    num_computed = getattr(common, "private_circle_num_computed", None)
    num_prompt = getattr(common, "private_circle_num_prompt", None)
    qsl = getattr(common, "query_start_loc_cpu", None)
    if qsl is None:
        qsl = getattr(common, "query_start_loc", None)
    qsl_values = qsl.tolist() if qsl is not None else None
    desc = []
    for i in range(num_reqs):
        parts = [
            f"row={i}",
            f"req={req_ids[i] if i < len(req_ids) else '?'}",
            f"prefill={bool(flags[i])}",
        ]
        if qsl_values is not None and i + 1 < len(qsl_values):
            parts.append(f"q={qsl_values[i + 1] - qsl_values[i]}")
        if num_computed is not None and i < len(num_computed):
            parts.append(f"c={num_computed[i]}")
        if num_prompt is not None and i < len(num_prompt):
            parts.append(f"p={num_prompt[i]}")
        desc.append(":".join(parts))
    return desc


def _assert_decode_rows_first(common: Any, num_reqs: int, num_decodes: int) -> None:
    """The positional split assumes decode rows precede prefill rows.

    vLLM does not guarantee this ordering; it holds under the default
    scheduler's admission dynamics but can break with
    long_prefill_token_threshold or the reordering schedulers.
    """
    is_prefilling = getattr(common, "is_prefilling", None)
    if (
        is_prefilling is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return
    flags = is_prefilling[:num_reqs].bool()
    if bool(flags[:num_decodes].any()):
        rows = [i for i in range(num_decodes) if bool(flags[i])]
        logger.error(
            "PRIVATE_CIRCLE_POOL batch_composition num_reqs=%d num_decodes=%d "
            "rows=[%s]",
            num_reqs,
            num_decodes,
            ", ".join(_private_circle_batch_rows(common, num_reqs, flags)),
        )
        raise RuntimeError(
            "V4.1 private circle pool requires decode rows before prefill "
            f"rows in the batch; prefill rows {rows[:8]} precede decode rows "
            "(see PRIVATE_CIRCLE_POOL batch_composition above; q=1 marks "
            "last-token-recompute rows, large q marks multi-step local "
            "prefill, '?' rows lack runner metadata). Disable "
            "long_prefill_token_threshold and the reordering schedulers "
            "(short_request_first / dyntra_lb / batch_job_aware / "
            "profiling_chunk / recompute) when the private circle pool is "
            "enabled."
        )


def _prepare_private_circle_plan(
    common: Any,
    shared: dict,
    vllm_config: VllmConfig,
    storage_block_size: int,
    blocks_per_allocation: int,
    num_real_reqs: int | None = None,
) -> dict:
    """Step-local compact workspace plan for private-circle Prefill.

    Prefill chunks longer than the ring would overwrite ring pages before FA
    reads them, so prefill rows execute from a shared compact workspace:
    restore the trailing history, write this step's KV, run FA, then commit
    only the trailing window to the private ring. Computed once per step and
    shared by every SWA layer; restore and commit run per layer. Host-only
    logic (.tolist()): the runner forces eager on these steps.
    """
    cached = shared.get("private_circle_plan")
    if cached is not None:
        return cached
    window_size = int(_config_value(vllm_config.model_config.hf_text_config, "sliding_window", 0))
    block_size = storage_block_size
    qsl = common.query_start_loc
    qsl_values = qsl.tolist()
    seq_values = common.seq_lens.tolist()
    num_reqs = len(seq_values)
    if num_real_reqs is not None:
        # common.seq_lens carries padded rows on graph/DP-aligned steps;
        # padding rows hold no allocations and must not enter the plan.
        num_reqs = min(num_reqs, int(num_real_reqs))
    query_lens = [qsl_values[i + 1] - qsl_values[i] for i in range(num_reqs)]
    num_decodes, num_decode_tokens, _num_prefills, _num_prefill_tokens = _request_counts(common, num_reqs)
    _assert_decode_rows_first(common, num_reqs, num_decodes)
    draft_tokens = (
        getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
        if vllm_config.speculative_config is not None
        else 0
    )
    capacity = window_size - 1 + 1 + draft_tokens
    if blocks_per_allocation != (capacity + block_size - 1) // block_size:
        raise RuntimeError(
            "private circle blocks_per_allocation disagrees with the pool "
            f"layout: impl={(capacity + block_size - 1) // block_size}, "
            f"pool={blocks_per_allocation}."
        )

    bounded_replay_rows: set[int] = set()
    bounded_replay_start = getattr(common, "private_circle_bounded_replay_start", None)
    persistent_start = getattr(common, "private_circle_persistent_start", None)
    if bounded_replay_start is not None and persistent_start is not None:
        bounded_replay_values = bounded_replay_start.tolist()
        persistent_values = persistent_start.tolist()
        bounded_replay_rows = {
            i
            for i in range(min(len(bounded_replay_values), len(persistent_values)))
            if 0 <= bounded_replay_values[i] < persistent_values[i]
        }
    block_table_rows = common.block_table_tensor[:num_reqs].tolist()
    max_workspace_blocks = _prefill_workspace_blocks(vllm_config, block_size, window_size)

    page_bases: list[int] = []
    history_lens: list[int] = []
    allocation_bases: list[int] = []
    workspace_seq_lens: list[int] = []
    confirmed_lens: list[int] = []
    workspace_rows: list[int] = []
    next_page = 0
    for row in range(num_decodes, num_reqs):
        query_len = int(query_lens[row])
        seq_len = int(seq_values[row])
        confirmed = seq_len - query_len
        if confirmed < 0:
            raise RuntimeError(
                f"inconsistent V4.1 prefill positions: row={row}, "
                f"confirmed={confirmed}, query_len={query_len}, seq_len={seq_len}"
            )
        history_len = 0 if row in bounded_replay_rows else min(window_size - 1, confirmed)
        local_len = history_len + query_len
        pages = max(1, (local_len + block_size - 1) // block_size)
        private_block = next((int(b) for b in block_table_rows[row] if int(b) > 0), -1)
        if private_block < 1:
            raise RuntimeError(f"private circle block table row {row} has no real block")
        allocation_base = 1 + ((private_block - 1) // blocks_per_allocation) * blocks_per_allocation
        if next_page + pages > max_workspace_blocks:
            raise RuntimeError(
                "V4.1 private circle prefill workspace exhausted; batch "
                f"exceeds profiled capacity ({next_page + pages} > "
                f"{max_workspace_blocks} blocks)"
            )
        page_bases.append(next_page)
        history_lens.append(history_len)
        allocation_bases.append(allocation_base)
        workspace_seq_lens.append(local_len)
        confirmed_lens.append(confirmed)
        workspace_rows.append(row)
        next_page += pages

    device = common.seq_lens.device
    restore_src: list[int] = []
    restore_dst: list[int] = []
    current_dst: list[int] = []
    commit_src: list[int] = []
    commit_dst: list[int] = []
    max_pages_per_req = max(
        1, max(((n + block_size - 1) // block_size for n in workspace_seq_lens), default=1)
    )
    workspace_block_table = torch.zeros(
        (len(workspace_rows), max_pages_per_req), dtype=torch.int32, device=device
    )
    for sub_row, row in enumerate(workspace_rows):
        query_len = query_lens[row]
        seq_len = seq_values[row]
        confirmed = confirmed_lens[sub_row]
        history_len = history_lens[sub_row]
        page_base = page_bases[sub_row]
        allocation_base = allocation_bases[sub_row]
        local_len = workspace_seq_lens[sub_row]
        pages = max(1, (local_len + block_size - 1) // block_size)
        workspace_block_table[sub_row, :pages] = torch.arange(
            page_base, page_base + pages, dtype=torch.int32, device=device
        )
        if row not in bounded_replay_rows:
            for j, absolute_pos in enumerate(range(confirmed - history_len, confirmed)):
                ring_block = (absolute_pos // block_size) % blocks_per_allocation
                restore_src.append((allocation_base + ring_block) * block_size + absolute_pos % block_size)
                restore_dst.append(page_base * block_size + j)
        current_dst.extend(page_base * block_size + history_len + j for j in range(query_len))
        keep = min(window_size, local_len)
        absolute_start = seq_len - keep
        local_start = local_len - keep
        for j in range(keep):
            absolute_pos = absolute_start + j
            ring_block = (absolute_pos // block_size) % blocks_per_allocation
            commit_src.append(page_base * block_size + local_start + j)
            commit_dst.append((allocation_base + ring_block) * block_size + absolute_pos % block_size)

    plan = {
        "vllm_config": vllm_config,
        "window_size": window_size,
        "block_size": block_size,
        "blocks_per_allocation": blocks_per_allocation,
        "num_decodes": num_decodes,
        "num_decode_tokens": int(num_decode_tokens),
        "workspace_num_blocks": max_workspace_blocks,
        "restore_src": torch.tensor(restore_src, dtype=torch.long, device=device),
        "restore_dst": torch.tensor(restore_dst, dtype=torch.long, device=device),
        "commit_src": torch.tensor(commit_src, dtype=torch.long, device=device),
        "commit_dst": torch.tensor(commit_dst, dtype=torch.long, device=device),
        "prefill_slots": _format_private_circle_slots(
            torch.tensor(current_dst, dtype=torch.int64, device=device), block_size
        ),
        "ws_block_table": workspace_block_table,
        "ws_seq_lens": torch.tensor(workspace_seq_lens, dtype=common.seq_lens.dtype, device=device),
        "prefill_query_start_loc": (qsl[num_decodes:] - int(qsl_values[num_decodes])).contiguous(),
        "ws_local_start_pos": torch.tensor(history_lens, dtype=torch.int32, device=device),
    }
    shared["private_circle_plan"] = plan
    return plan
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)



def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool
    cache_kind: str = "unknown"
    positions: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    start_pos: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_win_left: int = 0
    ori_win_right: int = 0
    smla_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None
    block_stride_rows: int = 0
    private_circle_plan: dict | None = None


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: DeepseekV41Metadata
    state: DeepseekV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: DeepseekV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: DeepseekV41Metadata | None
    swa: DeepseekV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None

    @property
    def positions(self) -> torch.Tensor:
        if self.swa.positions is None:
            raise RuntimeError("V4.1 SWA metadata does not contain input positions")
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        if self.swa.cos is None or self.swa.sin is None:
            raise RuntimeError("V4.1 SWA metadata does not contain RoPE tensors")
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _cache_coordinates(common: Any, ratio: int, compressed: bool):
    """Build the two coordinate vectors consumed by V4.1 cache operators.

    ``query_lens`` is derived locally only to compute ``start_pos``.  It is no
    longer part of the metadata contract: Indexer kernels use the cumulative
    query boundaries directly, avoiding a transient graph input tensor.
    """
    query_start_loc = common.query_start_loc[: common.num_reqs + 1]
    seq_lens = common.seq_lens[: common.num_reqs]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    start_pos = seq_lens - query_lens
    plane_ratio = ratio if compressed else 1
    cache_seq_lens = torch.div(seq_lens, plane_ratio, rounding_mode="floor")
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)

    return dict(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        start_pos=start_pos,
        cache_seq_lens=cache_seq_lens,
    )


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def _prefill_rows_fit_ring_capacity(
    common: Any,
    num_reqs: int,
    max_query_len: int,
) -> bool:
    """Whether every prefill row's query fits the ring's in-flight capacity.

    Rows within the per-step in-flight capacity (a PD request's
    last-token recompute, optionally fused with the first speculative
    draft block) write ring slots directly and need no shared prefill
    workspace, so their position in the batch does not matter.
    """
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return False
    flags = is_prefilling[:num_reqs].bool()
    starts = query_start_loc_cpu[: num_reqs + 1]
    query_lens = starts[1:] - starts[:-1]
    prefill_query_lens = query_lens[flags]
    return bool((prefill_query_lens <= max_query_len).all())


def _validate_batch_layout(
    common: Any,
    *,
    num_reqs: int,
    num_actual_reqs: int,
    num_input_tokens: int,
    num_actual_tokens: int,
) -> None:
    """Validate the shape contract shared by every V4.1 cache plane.

    The values themselves are device-side and must not be read back here. We
    only validate static lengths, preventing one plane from silently receiving
    a differently padded view of the same flattened batch.
    """
    if not 0 <= num_actual_reqs <= num_reqs:
        raise ValueError(
            "V4.1 metadata has invalid request counts: "
            f"actual={num_actual_reqs}, padded={num_reqs}"
        )
    if not 0 <= num_actual_tokens <= num_input_tokens:
        raise ValueError(
            "V4.1 metadata has invalid token counts: "
            f"actual={num_actual_tokens}, input={num_input_tokens}"
        )
    query_start_loc = getattr(common, "query_start_loc", None)
    block_table = getattr(common, "block_table_tensor", None)
    seq_lens = getattr(common, "seq_lens", None)
    slot_mapping = getattr(common, "slot_mapping", None)
    positions = getattr(common, "positions", None)
    if query_start_loc is None or query_start_loc.ndim != 1 or query_start_loc.numel() < num_reqs + 1:
        raise ValueError("V4.1 metadata requires query_start_loc with num_reqs + 1 entries")
    if block_table is None or block_table.ndim != 2:
        raise ValueError("V4.1 metadata requires a rank-2 block table")
    missing_block_rows = num_reqs - block_table.shape[0]
    if missing_block_rows > 1 or (missing_block_rows == 1 and num_actual_reqs != num_reqs - 1):
        raise ValueError("V4.1 metadata requires a block table row for every padded request")
    if seq_lens is None or seq_lens.ndim != 1 or seq_lens.numel() < num_reqs:
        raise ValueError("V4.1 metadata requires seq_lens for every padded request")
    if slot_mapping is None or slot_mapping.ndim != 1 or slot_mapping.numel() < num_input_tokens:
        raise ValueError("V4.1 metadata requires a slot for every input token")
    if positions is not None and (positions.ndim != 1 or positions.numel() < num_input_tokens):
        raise ValueError("V4.1 metadata requires a position for every input token")


def scatter_cache_v2(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and V4's Ascend op.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    the plane shape. ``npu_scatter_nd_update_v2`` preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        raise ValueError(
            f"V4.1 fused cache store requires builder-prepared [T, 2] slot_mapping, got {tuple(slot_mapping.shape)}"
        )
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]]
    updates = values.to(cache.dtype).contiguous()
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, indices, updates)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}")
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by fused Ascend operators.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def __init__(self, prefix, role, topology, long_kv_source_prefix, index_k_source_prefix):
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
            index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
            compressor_state = (
                metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing V4.1 cache metadata for {exc.args[0]}") from exc
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
        )

    @staticmethod
    def _reshape_query_heads(q: torch.Tensor, head_dim: int) -> torch.Tensor:
        if head_dim <= 0 or q.shape[-1] % head_dim:
            raise ValueError(
                "V4.1 query projection width must be divisible by head_dim: "
                f"width={q.shape[-1]}, head_dim={head_dim}"
            )
        return q.unflatten(-1, (q.shape[-1] // head_dim, head_dim))

    @staticmethod
    def _project_q_kv(attn, hidden_states, cos, sin):
        q_a = attn.wq_a(hidden_states)
        qr = attn.q_norm(q_a)
        q = attn.wq_b(qr)
        q = DeepseekV41EagerAttentionImpl._reshape_query_heads(q, attn.head_dim)
        kv = attn.kv_norm(attn.wkv(hidden_states))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        kv = kv.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr, kv.squeeze(1)


    def _to_attention_cache_order(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model [NoPE|RoPE] rows to the A5 cache [RoPE|NoPE] ABI."""
        if x.shape[-1] != 448 + 64:
            raise ValueError(
                f"A5 attention cache requires logical width 512, got {x.shape[-1]}"
            )

        return torch.cat(
            (
                x[..., 448:],   # RoPE：最后64维
                x[..., :448],   # NoPE：前448维
            ),
            dim=-1,
        )
    def _scatter_rows(self, cache: torch.Tensor, slot_mapping: torch.Tensor, rows: torch.Tensor) -> None:
        """Scatter packed rows into a possibly page-strided A3 slot view.

        Production metadata supplies ``[T, 2]`` page/offset coordinates and uses
        ``[-1, -1]`` for padded rows.  Keep that fixed-shape representation on NPU
        and delegate the skip/update semantics to the same ScatterNd primitive as
        the A3 index writer.  In particular, do not boolean-index device tensors:
        that lowers to dynamic ``Nonzero`` and cannot be captured by ACLGraph.
        """
        target = cache.squeeze(-2)
        slots = slot_mapping[: rows.shape[0]]
        if slots.ndim == 1:
            valid = slots >= 0
            physical = slots.clamp_min(0).long()
            pages = torch.div(physical, target.shape[1], rounding_mode="floor")
            offsets = physical.remainder(target.shape[1])
            indices = torch.stack((pages, offsets), dim=-1)
        elif slots.ndim == 2 and slots.shape[-1] == 2:
            valid = (slots >= 0).all(-1)
            indices = slots
        else:
            raise ValueError(f"A5 slot_mapping must be [T] or [T,2], got {tuple(slots.shape)}")

        # Normalize partially invalid coordinates as well as flat -1 slots to the
        # sentinel consumed by ScatterNd.  All tensors retain their static T shape.
        # CANN ScatterNd silently stops updating when an INT32-indexed target view
        # reaches 2 GiB.  V4.1's layer-outermost slots legitimately exceed that
        # size (the FP32 state ring gives the first three slots a 128 KiB page
        # stride), so use the operator's INT64 index path.  The page/row values are
        # still small; INT64 is required for the internal byte-offset calculation.
        indices = (
            torch.where(valid.unsqueeze(-1), indices, torch.full_like(indices, -1))
            .to(torch.int64)
            .contiguous()
        )
        updates = rows.to(target.dtype).contiguous()
        if target.device.type == "npu":
            torch_npu.npu_scatter_nd_update_(target, indices, updates)
            return

        # CPU-only deterministic golden used by unit tests.  Device execution must
        # stay on the fixed-shape branch above.
        if bool(valid.any()):
            target[indices[valid, 0].long(), indices[valid, 1].long()] = updates[valid]

    def _native_pack_rows(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        import custom_ops  # noqa: F401  # Registers torch.ops.custom.* from the wheel.

        if kind == "cmp":
            row_bytes, row_dtype, group_size, quant_mode = (
                320,
                torch.uint8,
                16,
                "mxfp4_bf16",
            )
        elif kind == "win":
            # The Host ABI validates FP8 cache dtype even though vLLM stores this
            # physical plane as opaque bytes.  Reinterpret the completed row as
            # U8 before scattering it into the shared 544-byte cache slot.
            row_bytes, row_dtype, group_size, quant_mode = (
                544,
                torch.float8_e4m3fn,
                32,
                "mxfp8_bf16",
            )
        else:
            raise ValueError(f"Unsupported A5 packed cache kind: {kind}")
        # V2 addresses cache as a flat row table.  Use local flat slots here and
        # scatter into vLLM's possibly page-strided cache only after the operator
        # has produced rows.  The attention ABI stores RoPE before NoPE.
        rows = torch.empty((x.shape[0], row_bytes), dtype=row_dtype, device=x.device)
        slots = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
        torch.ops.custom.kv_compress_epilog_v2(
            rows,
            x,
            slots,
            quant_group_size=group_size,
            quant_mode=quant_mode,
            round_scale=True,
            x_scale=1.0,
        )
        return rows.view(torch.uint8)

    def _write_attention_cache(
            self,
            cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            values: torch.Tensor,
            *,
            kind: str,
            backend: str = "reference",
    ) -> str:
        """Write cmp/win rows and return the backend that actually executed."""
        if backend not in ("reference", "auto", "native"):
            raise ValueError(f"Unsupported A5 cache-writer backend: {backend}")
        if values.shape[0] == 0:
            return "reference" if backend == "reference" else backend
        if backend != "reference":
            try:
                # 小算子实现
                # rows = self._native_pack_rows(values, kind)
                # self._scatter_rows(cache, slot_mapping, rows)
                # 融合算子实现
                slot = (slot_mapping[:, 0]) * cache.shape[1] + slot_mapping[:, 1]
                slot_mapping = slot.clamp(min=-1).to(torch.int32)
                if kind == "cmp":
                    group_size, quant_mode = (16, "mxfp4_bf16",)
                elif kind == "win":
                    group_size, quant_mode = (32, "mxfp8_bf16",)
                    # 算子校验要求
                    cache = cache.view(torch.float8_e4m3fn)
                else:
                    raise ValueError(f"Unsupported A5 packed cache kind: {kind}")
                torch.ops.custom.kv_compress_epilog_v2(
                cache,
                values,
                slot_mapping,
                quant_group_size=group_size,
                quant_mode=quant_mode,
                round_scale=True,
                x_scale=1.0,
                )
                return "native"
            except Exception:
                if backend == "native":
                    raise
        return "reference"

    def _get_private_circle_workspace(self, plan: dict, cache: torch.Tensor) -> torch.Tensor:
        return _get_private_circle_workspace(
            plan["vllm_config"],
            cache.dtype,
            cache.device,
            cache.shape[1:],
            plan["workspace_num_blocks"],
        )

    def _scatter_swa_kv(self, attn, kv: torch.Tensor, swa_metadata) -> None:
        """Write this step's KV: decode rows to the ring, prefill to workspace.

        The pool decides WHERE (ring/workspace); the base `_write_attention_cache`
        decides HOW (A5 quantized pack).
        """
        cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        plan = getattr(swa_metadata, "private_circle_plan", None)
        if plan is None:
            self._write_attention_cache(
                cache, swa_metadata.slot_mapping, kv, kind="win", backend="native"
            )
            return
        workspace = self._get_private_circle_workspace(plan, cache)
        restore_src = plan["restore_src"]
        if restore_src.numel() > 0:
            # Byte views: lossless for low-precision caches.
            workspace.flatten(0, 1).view(torch.uint8).index_copy_(
                0,
                plan["restore_dst"],
                cache.flatten(0, 1).view(torch.uint8).index_select(0, restore_src),
            )
        num_decode_tokens = plan["num_decode_tokens"]
        if num_decode_tokens:
            self._write_attention_cache(
                cache,
                swa_metadata.slot_mapping[:num_decode_tokens],
                kv[:num_decode_tokens],
                kind="win",
                backend="native",
            )
        if kv.shape[0] > num_decode_tokens:
            self._write_attention_cache(
                workspace,
                swa_metadata.slot_mapping[num_decode_tokens:],
                kv[num_decode_tokens:],
                kind="win",
                backend="native",
            )

    def _commit_private_circle_prefill(self, plan: dict, cache: torch.Tensor, workspace: torch.Tensor) -> None:
        """Commit only the post-chunk tail window to the private ring."""
        commit_src = plan["commit_src"]
        if commit_src.numel() == 0:
            return
        cache.flatten(0, 1).view(torch.uint8).index_copy_(
            0,
            plan["commit_dst"],
            workspace.flatten(0, 1).view(torch.uint8).index_select(0, commit_src),
        )

    def preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Project Q/KV and populate this layer's SWA cache on the current stream."""
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        self._scatter_swa_kv(attn, kv, swa_metadata)
        return q, qr

    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        share_quant = (
            type(wq_a._quant_method) is type(wkv._quant_method) and wq_a._has_communication == wkv._has_communication
        )

        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm, RoPE and cache store (Vector).
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            self._scatter_swa_kv(attn, kv.squeeze(1), swa_metadata)
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias)
        q = self._reshape_query_heads(q, attn.head_dim)
        main_stream.wait_stream(aux_stream)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr

    def _write_compressed_source(
        self,
        attn,
        hidden_states,
        positions,
        cos,
        sin,
        metadata,
        *,
        fxrt_decomposed: bool = False,
    ):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_metadata = compressor_metadata.state
            if state_metadata.c2_ring_metadata is None or state_metadata.c2_metadata_group_id is None:
                raise RuntimeError("V4.1 ring compressor metadata is missing")
            if fxrt_decomposed:
                from vllm_ascend.ops.dsv41_prefill import wait_metadata

                wait_metadata(int(DeviceMetadataStage.COMPRESSOR), state_metadata.c2_metadata_group_id)
            else:
                wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            latent = compressor.pool_projected(hidden_states, state_metadata)
            source_cos = state_metadata.c2_source_cos
            source_sin = state_metadata.c2_source_sin
            if source_cos is None or source_sin is None:
                fallback_cos, fallback_sin = get_cos_and_sin_dsa(state_metadata.c2_source_positions)
                source_cos = fallback_cos[attn.rotary_emb.layername]
                source_sin = fallback_sin[attn.rotary_emb.layername]
            source_cos = source_cos[: positions.shape[0]]
            source_sin = source_sin[: positions.shape[0]]
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]

        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            latent.unsqueeze(1),
            source_cos,
            source_sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        if attn.long_kv_cache.kv_cache[0].dtype == torch.uint8:
            # A5 quantized path: the epilog op packs mxfp4 rows into the
            # flat uint8 view and expects the RoPE dims first.
            try:
                import custom_ops  # noqa: F401  registers torch.ops.custom.*
            except ImportError as exc:
                raise RuntimeError(
                    "DeepSeek V4.1 compressed-KV store requires the custom_ops module "
                    "registering torch.ops.custom.kv_compress_epilog_v2."
                ) from exc
            cache_4d = attn.long_kv_cache.kv_cache[0]
            self._write_attention_cache(cache_4d,
                long_slots,
                latent.squeeze(1),
                kind="cmp",
                backend="native"
            )
        else:
            scatter_cache_v2(
                attn.long_kv_cache.kv_cache[0],
                long_slots,
                latent.squeeze(1),
            )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            return shared.topk_indices[: hidden_states.shape[0]]
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        selected, candidate_indices, candidate_lengths = attn.indexer.select(
            hidden_states,
            qr,
            positions,
            cos,
            sin,
            source_layer.kv_cache[0],
            metadata.indexer.cache,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidate_indices=shared.candidate_indices[: hidden_states.shape[0]],
            candidate_lengths=shared.candidate_lengths[: hidden_states.shape[0]],
        )
        shared.topk_indices[: selected.shape[0]].copy_(selected)
        if self.role.is_candidate_source:
            shared.candidate_indices[: candidate_indices.shape[0]].copy_(candidate_indices)
            shared.candidate_lengths[: candidate_indices.shape[0]].copy_(
                candidate_lengths.reshape(candidate_indices.shape[0], -1)[:, :1]
            )
        return shared.topk_indices[: selected.shape[0]]

    def _attention(self, attn, q, metadata, compressed_indices):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            compressed_indices=compressed_indices,
        )


    def _get_window_topk_idxs(
            self,
            attn,
            q: torch.Tensor,
            swa_metadata: DeepseekV41Metadata,
            num_reqs: int,
            query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sliding-window KV slot indices in TNK format.

        Fully vectorised on-device implementation — no per-request Python
        loop, no CPU-side tensor creation, no D2H/H2D copy.

        Returns a ``[total_tokens, 1, window_size]`` int32 tensor
        where rows are concatenated across requests (TNK layout).  ``-1`` marks
        a slot that holds nothing or padding.  Returns all ``-1`` when
        ``start_pos`` is unavailable (e.g. drafting metadata).
        """
        window_size = attn.window_size
        total_tokens = q.shape[0]
        device = q.device
        start_pos = swa_metadata.start_pos
        if start_pos is None or total_tokens == 0:
            return torch.full(
                (total_tokens, 1, window_size),
                -1,
                dtype=torch.int32,
                device=device,
            )
        global_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
        token_req_idx = torch.searchsorted(
            query_start_loc[1:num_reqs + 1], global_indices, right=True
        )
        token_pos = global_indices - query_start_loc[token_req_idx].to(torch.int32)
        token_sp = start_pos[token_req_idx].to(torch.int32)
        cols = torch.arange(window_size, device=device, dtype=torch.int32)
        token_abs_pos = token_sp + token_pos
        window_start = torch.clamp(token_abs_pos - window_size + 1, min=0)
        win_indices = window_start.unsqueeze(1) + cols.unsqueeze(0)
        win_indices = win_indices.masked_fill(
            win_indices > token_abs_pos.unsqueeze(1), -1
        )
        return win_indices.unsqueeze(1).to(torch.int32)

    def _prepare_cmp_attention(
        self,
        metadata,
        source_cache,
        compressed_indices,
        num_reqs: int,
    ):
        """Shared compressed-KV preparation for the sparse MLA stages."""
        if self.role.compress_ratio not in (1, 2):
            return None, None, None
        if source_cache is None or metadata.attention is None or compressed_indices is None:
            raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
        cmp_block_table = metadata.attention.block_table[:num_reqs]
        cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
        cmp_topk = self.topology.index_topk
        if cmp_topk not in (512, 1024):
            raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
        return pad_sparse_indices(compressed_indices, cmp_topk), cmp_block_table, cmp_seq_lens

    def _private_circle_attention(
        self,
        attn,
        q,
        metadata,
        plan: dict,
        source_cache,
        compressed_indices,
    ):
        """Mixed batches split at num_decodes: decode rows run on the ring
        with absolute seq_lens, prefill rows on the compact workspace so long
        chunks never overwrite ring pages before FA reads them."""
        has_compressed = self.role.compress_ratio in (1, 2)
        num_reqs = metadata.swa.num_reqs
        num_decodes = plan["num_decodes"]
        num_decode_tokens = min(plan["num_decode_tokens"], q.shape[0])
        cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        workspace = self._get_private_circle_workspace(plan, cache)

        cmp_indices, cmp_block_table, cmp_seq_lens = self._prepare_cmp_attention(
            metadata, source_cache, compressed_indices, num_reqs
        )
        if cmp_indices is not None:
            decode_cmp_indices = cmp_indices[:num_decode_tokens]
            decode_cmp_block_table = cmp_block_table[:num_decodes]
            decode_cmp_seq_lens = cmp_seq_lens[:num_decodes]
            prefill_cmp_indices = cmp_indices[num_decode_tokens:]
            prefill_cmp_block_table = cmp_block_table[num_decodes:]
            prefill_cmp_seq_lens = cmp_seq_lens[num_decodes:]
        else:
            decode_cmp_indices = decode_cmp_block_table = decode_cmp_seq_lens = None
            prefill_cmp_indices = prefill_cmp_block_table = prefill_cmp_seq_lens = None

        outputs = []
        if num_decodes:
            if metadata.swa.start_pos is None:
                raise RuntimeError(
                    "V4.1 private circle decode is missing SWA start_pos metadata"
                )
            decode_qsl = metadata.swa.query_start_loc[: num_decodes + 1]
            decode_start_pos = metadata.swa.start_pos[:num_decodes]
            outputs.append(
                self._sparse_mla_attention(
                    attn,
                    q[:num_decode_tokens],
                    win_kv=cache,
                    win_block_table=metadata.swa.block_table[:num_decodes],
                    cu_seqlens_q=decode_qsl,
                    seqused_win_kv=metadata.swa.seq_lens[:num_decodes],
                    win_indices=self._get_window_topk_idxs(
                        attn,
                        q[:num_decode_tokens],
                        SimpleNamespace(start_pos=decode_start_pos),
                        num_decodes,
                        decode_qsl,
                    ),
                    cmp_indices=decode_cmp_indices,
                    cmp_block_table=decode_cmp_block_table,
                    seqused_cmp_kv=decode_cmp_seq_lens,
                    source_cache=source_cache,
                    has_compressed=has_compressed,
                )
            )
        if q.shape[0] > num_decode_tokens:
            prefill_q = q[num_decode_tokens:]
            outputs.append(
                self._sparse_mla_attention(
                    attn,
                    prefill_q,
                    win_kv=workspace,
                    win_block_table=plan["ws_block_table"],
                    cu_seqlens_q=plan["prefill_query_start_loc"],
                    seqused_win_kv=plan["ws_seq_lens"],
                    win_indices=self._get_window_topk_idxs(
                        attn,
                        prefill_q,
                        SimpleNamespace(start_pos=plan["ws_local_start_pos"]),
                        num_reqs - num_decodes,
                        plan["prefill_query_start_loc"],
                    ),
                    cmp_indices=prefill_cmp_indices,
                    cmp_block_table=prefill_cmp_block_table,
                    seqused_cmp_kv=prefill_cmp_seq_lens,
                    source_cache=source_cache,
                    has_compressed=has_compressed,
                )
            )
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        self._commit_private_circle_prefill(plan, cache, workspace)
        return output

    def _sparse_mla_attention(
        self,
        attn,
        q: torch.Tensor,
        *,
        win_kv: torch.Tensor,
        win_block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_win_kv: torch.Tensor,
        win_indices: torch.Tensor,
        cmp_indices: torch.Tensor | None,
        cmp_block_table: torch.Tensor | None,
        seqused_cmp_kv: torch.Tensor | None,
        source_cache,
        has_compressed: bool,
    ) -> torch.Tensor:
        """One mixed_quant_sparse_flash_mla call: shared by the base full-batch
        path and the private-circle decode/prefill sub-batches."""
        win_topk_length = (win_indices >= 0).sum(dim=-1).to(torch.int32)
        cmp_topk_length = (
            torch.zeros_like(win_topk_length)
            if cmp_indices is None
            else (cmp_indices >= 0).sum(dim=-1).to(torch.int32)
        )
        op_metadata = mixed_quant_sparse_flash_mla_metadata(
            win_topk_length,
            cmp_topk_length,
            cu_seqlens_q=cu_seqlens_q,
            num_heads_q=q.shape[1],
            num_heads_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0].shape[2],
            head_dim=q.shape[-1],
            quant_mode=1,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_win_kv=True,
            has_cmp_kv=has_compressed,
        )
        cmp_topk_length = None if cmp_indices is None else cmp_topk_length
        output, _ = cann_ops_transformer.ops.ds41.mixed_quant_sparse_flash_mla(
            q,
            win_kv=win_kv,
            cmp_kv=source_cache,
            win_sparse_indices=win_indices,
            cmp_sparse_indices=cmp_indices,
            win_block_table=win_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_win_kv=seqused_win_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            win_topk_length=win_topk_length,
            cmp_topk_length=cmp_topk_length,
            sinks=attn.attn_sink.data,
            metadata=op_metadata,
            quant_mode=1,
            softmax_scale=attn.softmax_scale,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        )
        return output

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        compressed_indices,
    ):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        if attn.head_dim != 512:
            raise ValueError(f"SparseFlashMla requires head_dim 512, got {attn.head_dim}")
        if attn.window_size != 128:
            raise ValueError(f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}")
        if not 1 <= attn.n_local_heads <= 128 or attn.n_local_heads & (attn.n_local_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {attn.n_local_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        private_circle_plan = getattr(metadata.swa, "private_circle_plan", None)
        if private_circle_plan is not None:
            return self._private_circle_attention(
                attn, q, metadata, private_circle_plan, source_cache, compressed_indices
            )
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        cmp_indices, cmp_block_table, cmp_seq_lens = self._prepare_cmp_attention(
            metadata, source_cache, compressed_indices, num_reqs
        )
        return self._sparse_mla_attention(
            attn,
            q,
            win_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
            win_block_table=metadata.swa.block_table[:num_reqs],
            cu_seqlens_q=query_start_loc,
            seqused_win_kv=metadata.swa.seq_lens[:num_reqs],
            win_indices=self._get_window_topk_idxs(
                attn, q, metadata.swa, num_reqs, query_start_loc
            ),
            cmp_indices=cmp_indices,
            cmp_block_table=cmp_block_table,
            seqused_cmp_kv=cmp_seq_lens,
            source_cache=source_cache,
            has_compressed=has_compressed,
        )


    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def _project_output(self, attn, attention_output, hidden_states, *, projected):
        """Write the O-projection result into a caller-supplied buffer.
        When the attention output has fewer rows than ``hidden_states``
        (e.g. graph padding), zero-pad before the matmul so the stale tail
        of ``projected`` is never returned to the model.
        """
        padded = attention_output
        if attention_output.shape[0] != hidden_states.shape[0]:
            padded = attention_output.new_zeros(
                (hidden_states.shape[0], attention_output.shape[1], attention_output.shape[2])
            )
            padded[: attention_output.shape[0]] = attention_output
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(padded, projected)
        return projected

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None, *, fxrt_decomposed=False):
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            output.zero_()
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        positions = metadata.positions[: hidden_states.shape[0]]
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        v1_impl = attn.dsa_attn.dsa_attn.impl
        if fxrt_decomposed and v1_impl.multistream_dsv4_dsa_overlap:
            from vllm_ascend.ops.dsv41_prefill import (
                prolog_q_a,
                prolog_q_b,
                prolog_q_norm,
            )

            q_a, kv_quant, kv_scale = prolog_q_a(
                hidden_states, attn.v41_layer_name
            )
            qr, q_b_quant, q_b_scale, kv = prolog_q_norm(
                hidden_states,
                q_a,
                kv_quant,
                kv_scale,
                attn.v41_layer_name,
            )
            cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
            q = prolog_q_b(
                hidden_states,
                qr,
                q_b_quant,
                q_b_scale,
                kv,
                cos,
                sin,
                [cache],
                attn.v41_layer_name,
            )
        else:
            preprocess = self.multistream_preprocess if v1_impl.multistream_dsv4_dsa_overlap else self.preprocess
            q, qr = preprocess(attn, hidden_states, cos, sin, metadata.swa)
        if self.role.is_kv_source:
            self._write_compressed_source(
                attn,
                hidden_states,
                positions,
                cos,
                sin,
                metadata,
                fxrt_decomposed=fxrt_decomposed,
            )
        compressed_indices = self._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)
        attention_output = self._attention(attn, q, metadata, compressed_indices)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            attention_output.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        self._project_output(attn, attention_output, hidden_states, projected=output)
        return output


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        # Metadata consumed from inside an ACLGraph must not point at a
        # per-step cast/advanced-index result. Keep positions in a stable
        # builder-owned buffer just like the cache coordinates below.
        self._positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._block_table: torch.Tensor | None = None
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._start_pos = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._smla_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._c2_ring_metadata = torch.zeros(5 * max_reqs, dtype=torch.int32, device=device)
        self._c2_complete_mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self._c2_source_positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        text_config = vllm_config.model_config.hf_text_config
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = (
            max_tokens if self._supports_device_ops and isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec) else 0
        )
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> DeepseekV41Metadata:
        # Direct callers of the capture hook do not necessarily pass the
        # runtime-mode keyword used by model_runner. Capture must nevertheless
        # receive stable RoPE buffers, exactly like FULL replay.
        kwargs.setdefault("full_graph_mode", True)
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if isinstance(self.kv_cache_spec, DeepseekV41CompressorStateSpec):
            if not self._c2_rope_layer_names:
                raise RuntimeError("V4.1 compressor-state builder has no source RoPE layer")
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            for rope_layer_name in self._c2_rope_layer_names[1:]:
                other_rope = get_full_cos_and_sin_dsa_for_layer(rope_layer_name)
                if any(other.data_ptr() != source.data_ptr() for other, source in zip(other_rope, source_rope)):
                    raise RuntimeError("V4.1 ratio-2 source layers must share one RoPE table")
            self._c2_full_source_rope = source_rope

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled:
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer)),
            )
        else:
            run()
        return buffer

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        if not is_v41_draft_swa_spec(self.kv_cache_spec):
            raise TypeError("V4.1 drafting requires a draft SWA cache")
        # DSpark issues one eager block per step. Group-local tables and slots
        # remain independent; the builder owns the operator metadata buffers.
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **kwargs,
    ):
        # Prefix blocks are already represented by block_table and seq_lens.
        # The V4.1 operators do not consume a separate common-prefix length,
        # so keep the same semantics as DSA v1 and ignore this hint.
        self._device_metadata_tasks = ()
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(spec, (DeepseekV41SWASpec, DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec)):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = int(kwargs.get("num_actual_reqs", num_reqs))
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        num_actual_tokens = int(getattr(common, "num_actual_tokens", num_input_tokens))
        _validate_batch_layout(
            common,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_input_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
        )
        shared = kwargs.get("common_v41_metadata")
        if shared is None:
            shared = {}

        # Keep a complete request-dimension table at a stable address. A
        # mixed full-graph batch can contain one synthetic padding request
        # whose row is not present in the runner-owned table.
        source_block_table = common.block_table_tensor
        if self._block_table is None or self._block_table.shape[1] != source_block_table.shape[1]:
            self._block_table = torch.zeros(
                (self._seq_lens.shape[0] + 1, source_block_table.shape[1]),
                dtype=source_block_table.dtype,
                device=source_block_table.device,
            )
        if num_reqs > self._block_table.shape[0]:
            raise ValueError(
                "V4.1 block-table buffer is smaller than the padded request count: "
                f"{num_reqs} > {self._block_table.shape[0]}"
            )
        self._block_table[:num_reqs].zero_()
        rows_to_copy = min(num_reqs, source_block_table.shape[0])
        if rows_to_copy:
            self._block_table[:rows_to_copy].copy_(source_block_table[:rows_to_copy])
        block_table = self._block_table[:num_reqs]

        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        raw_slots = (
            common.slot_mapping
            if cache_kind == "swa"
            else torch.full_like(common.slot_mapping, -1)
            if cache_kind == "compressor_state"
            else compressed_slot_mapping(common.slot_mapping, ratio)
        )
        if is_compressor_state:
            if self._supports_device_ops:
                self._slot_mapping[:num_input_tokens].copy_(raw_slots[:num_input_tokens])
                slots = self._slot_mapping[:num_input_tokens]
            else:
                # State writes use ring ownership metadata; ordinary slots stay PAD.
                slots = raw_slots
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout then share one persistent [T, 2] mapping, while every SWA
            # group owns a distinct mapping buffer.
            mapping_scope = "swa" if cache_kind == "swa" else "compressed"
            slot_key = f"slot:{mapping_scope}:c{ratio}:b{spec.storage_block_size}"
            prepared_slots = shared.get(slot_key)
            if prepared_slots is None:
                active_slots = raw_slots[:num_input_tokens]
                valid = active_slots >= 0
                if compressed and ratio == 2:
                    # Prepare the C2 store mask once per cache group, before
                    # forward. Match the ring compressor's completion policy.
                    if kwargs.get("skip_ring_state_update", False):
                        valid.zero_()
                    else:
                        valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                        valid &= torch.arange(num_input_tokens, device=active_slots.device) < valid_end
                        if common.positions is not None:
                            valid &= common.positions[:num_input_tokens].remainder(2) == 1
                physical = active_slots.clamp_min(0)
                self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                    torch.where(
                        valid,
                        torch.div(
                            physical,
                            spec.storage_block_size,
                            rounding_mode="floor",
                        ),
                        -1,
                    )
                )
                self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                    torch.where(
                        valid,
                        physical.remainder(spec.storage_block_size),
                        -1,
                    )
                )
                prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                shared[slot_key] = prepared_slots
            slots = prepared_slots
        coordinates = _cache_coordinates(common, ratio, compressed)
        self._seq_lens[:num_reqs].copy_(coordinates["seq_lens"])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        # start_pos must live in a persistent buffer: the attention forward
        # reads it inside the ACLGraph capture, and a per-step allocation
        # would bake a stale address into the replayed graph.
        self._start_pos[:num_reqs].copy_(coordinates["start_pos"])
        if num_actual_reqs < num_reqs:
            self._start_pos[num_actual_reqs:num_reqs].zero_()
        plane_ratio = ratio if compressed else 1
        self._cache_seq_lens[:num_reqs].copy_(
            torch.div(
                self._seq_lens[:num_reqs],
                plane_ratio,
                rounding_mode="floor",
            )
        )
        coordinates["seq_lens"] = self._seq_lens[:num_reqs]
        coordinates["start_pos"] = self._start_pos[:num_reqs]
        coordinates["cache_seq_lens"] = self._cache_seq_lens[:num_reqs]
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            self._cmp_residual[:num_reqs].copy_(self._seq_lens[:num_reqs].remainder(ratio))
            cmp_residual_buffer = self._cmp_residual[:num_reqs]
        positions = getattr(common, "positions", None)
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            self._positions[:num_input_tokens].copy_(positions[:num_input_tokens].to(torch.int64))
            positions = self._positions[:num_input_tokens]
        (
            num_decodes,
            num_decode_tokens,
            num_prefills,
            num_prefill_tokens,
        ) = _request_counts(common, num_actual_reqs)
        draft_tokens = (
            getattr(self.vllm_config.speculative_config, "num_speculative_tokens", 0)
            if self.vllm_config.speculative_config is not None
            else 0
        )
        if (
            cache_kind == "swa"
            and num_prefills > 0
            and not is_v41_draft_swa_spec(spec)
            and getattr(common, "private_circle_blocks_per_allocation", None)
            and _prefill_rows_fit_ring_capacity(
                common, num_actual_reqs, 1 + draft_tokens
            )
        ):
            # Every prefill row's query fits the ring's in-flight capacity:
            # a PD request's last-token recompute, optionally fused with the
            # first speculative draft block. The imported ring already covers
            # the window, so those rows run as decodes: no plan, and the step
            # stays graph-capturable regardless of row order.
            num_decodes += num_prefills
            num_decode_tokens += num_prefill_tokens
            num_prefills = 0
            num_prefill_tokens = 0
        if cache_kind == "swa" and positions is not None:
            # Single-token prefill rows (a PD request's last-token recompute)
            # appear in DecodeOnly graph steps. The fresh-indexing branch would
            # bake a per-step allocation into the replayed graph; those steps
            # must gather into the stable runtime buffers like pure decodes.
            # Their token count (num_reqs) is far below the buffer capacity.
            cos, sin = get_cos_and_sin_dsa(
                positions,
                use_cache=(
                    bool(kwargs.get("full_graph_mode", False))
                    or num_prefills == 0
                    or num_prefill_tokens == num_prefills
                ),
            )
        text_config = self.vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        n_local_heads = (
            int(_config_value(text_config, "num_attention_heads"))
            // self.vllm_config.parallel_config.tensor_parallel_size
        )
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        operator_ratio = 0 if cache_kind == "swa" else ratio
        smla_metadata = None
        qli_metadata = None
        private_circle_plan = None
        # Draft SWA layers live in the shared slots, not the request-private
        # ring: their builds must never construct a pool plan, or the plan's
        # ring-geometry restore/commit would run against the draft's
        # shared-slot cache.
        if (
            cache_kind == "swa"
            and num_prefills > 0
            and not is_v41_draft_swa_spec(spec)
        ):
            blocks_per_allocation = getattr(common, "private_circle_blocks_per_allocation", None)
            if blocks_per_allocation:
                private_circle_plan = _prepare_private_circle_plan(
                    common,
                    shared,
                    self.vllm_config,
                    spec.storage_block_size,
                    int(blocks_per_allocation),
                    num_real_reqs=num_actual_reqs,
                )
                # Prefill tokens scatter into the shared compact workspace;
                # decode tokens keep the ring slots from the group table.
                # Padding rows beyond the real requests contribute no slots.
                self._private_circle_slots_2d = torch.empty(
                    (num_input_tokens, 2), dtype=torch.int32, device=self.device
                )
                if num_decode_tokens:
                    ring_slots = _format_private_circle_slots(
                        common.slot_mapping[:num_decode_tokens].long(),
                        spec.storage_block_size,
                    )
                    self._private_circle_slots_2d[:num_decode_tokens].copy_(ring_slots)
                prefill_end = num_decode_tokens + num_prefill_tokens
                self._private_circle_slots_2d[num_decode_tokens:prefill_end].copy_(
                    private_circle_plan["prefill_slots"]
                )
                self._private_circle_slots_2d[prefill_end:].fill_(-1)
                slots = self._private_circle_slots_2d[:num_input_tokens]
# TODO kezong
        # if self._supports_device_ops and cache_kind in {"swa", "long_kv"}:
        #     has_compressed = operator_ratio in (1, 2)
        #     cmp_seq_lens = self._cache_seq_lens[:num_reqs] if has_compressed else None
        #     cmp_residual = cmp_residual_buffer

        #     def build_smla_metadata() -> None:
        #         value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        #             n_local_heads,
        #             1,
        #             head_dim,
        #             cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
        #             seqused_ori_kv=self._seq_lens[:num_reqs],
        #             seqused_cmp_kv=cmp_seq_lens,
        #             cmp_residual_kv=cmp_residual,
        #             batch_size=num_reqs,
        #             max_seqlen_q=int(getattr(common, "max_query_len", 0)),
        #             max_seqlen_ori_kv=int(getattr(common, "max_seq_len", 0)),
        #             max_seqlen_cmp_kv=0,
        #             ori_topk=0,
        #             cmp_topk=index_topk if has_compressed else 0,
        #             cmp_ratio=operator_ratio,
        #             ori_mask_mode=4,
        #             cmp_mask_mode=3 if has_compressed else 0,
        #             ori_win_left=max(0, window_size - 1),
        #             ori_win_right=0,
        #             layout_q="TND",
        #             layout_kv="PA_BBND",
        #             has_ori_kv=True,
        #             has_cmp_kv=has_compressed,
        #         )
        #         self._smla_metadata.copy_(value)

        #     smla_metadata = self._publish_task(
        #         shared,
        #         f"smla:c{operator_ratio}",
        #         self._smla_metadata,
        #         DeviceMetadataStage.ATTENTION,
        #         build_smla_metadata,
        #     )

        if self._supports_device_ops and cache_kind == "index_k":
            self._qli_metadata = None
            qli_metadata = None

        c2_ring_metadata = None
        c2_complete_mask = None
        c2_source_positions = None
        c2_source_cos = None
        c2_source_sin = None
        c2_metadata_group_id = None
        if cache_kind == "compressor_state" and getattr(common, "positions", None) is not None:
            ring_meta = self._c2_ring_metadata[: 5 * num_reqs].view(5, num_reqs)
            input_positions = common.positions[:num_input_tokens].long()
            if self._supports_device_ops:
                if self._c2_full_source_rope is None:
                    raise RuntimeError("V4.1 source RoPE buffers were not initialized")
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None

            def build_c2_metadata() -> None:
                starts = common.query_start_loc[:num_reqs].int()
                ends = common.query_start_loc[1 : num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(num_reqs, device=starts.device) < num_actual_reqs
                used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if kwargs.get("skip_ring_state_update", False):
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((self._seq_lens[:num_reqs] - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, block_table[:num_reqs, 0], 0))
                valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                valid = torch.arange(num_input_tokens, device=input_positions.device) < valid_end
                complete = (input_positions.remainder(2) == 1) & valid
                if kwargs.get("skip_ring_state_update", False):
                    complete = torch.zeros_like(complete)
                self._c2_complete_mask[:num_input_tokens].copy_(complete)
                self._c2_source_positions[:num_input_tokens].copy_(
                    torch.where(
                        complete,
                        input_positions - 1,
                        torch.zeros_like(input_positions),
                    )
                )
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        self._c2_source_positions[:num_input_tokens]
                        .reshape(-1, 1, 1, 1)
                        .expand(
                            num_input_tokens,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[:num_input_tokens],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[:num_input_tokens],
                    )

            compressor_group = self._publish_task(
                shared,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
            )
            if compressor_group is not self._c2_complete_mask:
                raise RuntimeError("V4.1 compressor metadata must have one owner")
            c2_complete_mask = self._c2_complete_mask[:num_input_tokens]
            c2_ring_metadata = ring_meta
            c2_source_positions = self._c2_source_positions[:num_input_tokens]
            if self._supports_device_ops:
                c2_source_cos = self._c2_source_cos[:num_input_tokens]
                c2_source_sin = self._c2_source_sin[:num_input_tokens]
            c2_metadata_group_id = id(self._c2_complete_mask)
        block_stride_rows = 0
        if cache_kind == "long_kv" and isinstance(spec, DeepseekV41FullSpec) and spec.dtype == torch.uint8:
            block_stride_rows = (getattr(spec, "page_size_padded", None) or 0) // spec.head_size
        return DeepseekV41Metadata(
            block_table=block_table,
            slot_mapping=slots,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            is_compressor_state=is_compressor_state,
            cache_kind=cache_kind,
            block_stride_rows=block_stride_rows,
            positions=positions,
            cos=cos,
            sin=sin,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_win_left=max(0, window_size - 1),
            ori_win_right=0,
            smla_metadata=smla_metadata,
            qli_metadata=qli_metadata,
            cmp_residual=cmp_residual_buffer,
            c2_ring_metadata=c2_ring_metadata,
            c2_complete_mask=c2_complete_mask,
            c2_source_positions=c2_source_positions,
            c2_source_cos=c2_source_cos,
            c2_source_sin=c2_source_sin,
            c2_metadata_group_id=c2_metadata_group_id,
            private_circle_plan=private_circle_plan,
            **coordinates,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        return DeepseekV41EagerAttentionImpl

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend
