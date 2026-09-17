# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_v2,
)
from vllm_ascend.core.deepseek_v41 import DeepseekV41IndexerSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .compressor import DeepseekV41RMSNorm, _read
from cann_ops_transformer.ops.ds41 import quant_lightning_indexer
from cann_ops_transformer.ops.ds41 import quant_sparse_lightning_indexer

class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.compress_ratio = compress_ratio
        self.n_heads = int(_read(config, "index_n_heads"))
        self.width = int(_read(config, "index_head_dim"))
        self.rope_width = int(_read(config, "qk_rope_head_dim"))
        self.index_topk = int(_read(config, "index_topk"))
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        self.wq_b = ReplicatedLinear(
            _read(config, "q_lora_rank"),
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            _read(config, "hidden_size"),
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                _read(config, "head_dim"),
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
            use_a5_quantized_cache = get_ascend_device_type() == AscendDeviceType.A5
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41IndexerSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width // 2 if use_a5_quantized_cache else self.width,
                    dtype=torch.uint8 if use_a5_quantized_cache else torch.int8,
                    compress_ratio=compress_ratio,
                    scale_dim=self.width // 32 if use_a5_quantized_cache else 1,
                    scale_dtype=(torch.uint8 if use_a5_quantized_cache else torch.float16),
                ),
            )

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        slot_mapping = (slots[:, 0]) * k_cache.shape[1] + slots[:, 1]
        slots = slot_mapping.clamp(min=-1).to(torch.int32)
        key = key.squeeze(1)

        torch.ops.cann_ops_transformer.indexer_quant_cache(
            cache=k_cache,
            cache_scale=scale_cache,
            x=key,
            slot_mapping=slots,
            quant_mode="mxfp4"
        )
        '''print("zzx update_keys k_cache", k_cache.shape, k_cache.dtype)
        print("zzx update_keys scale_cache", scale_cache.shape, scale_cache.dtype)
        print("zzx update_keys key", key.shape, key.dtype)'''

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_indices,
        candidate_lengths,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * self.weights_scale

        return self.select_projected(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
        )

    def select_projected(
        self,
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_indices,
        candidate_lengths,
    ):
        if is_candidate_source and uses_candidate_filter:
            raise ValueError("A candidate source must use the unfiltered position TopK")
        if uses_candidate_filter and (candidate_indices is None or candidate_lengths is None):
            raise RuntimeError("V4.1 candidate-filtering indexer ran before its source")
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidate_indices.shape != candidate_shape or candidate_indices.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
        topk = self.index_topk
        if query.shape[0] == 0:
            selected = torch.full(
                (0, topk), -1, dtype=torch.int32, device=query.device
            )
            if is_candidate_source:
                candidate_indices = torch.full(
                    candidate_shape, -1, dtype=torch.int32, device=query.device
                )
                candidate_lengths = torch.zeros(
                    (0, 1), dtype=torch.int32, device=query.device
                )
            return selected, candidate_indices, candidate_lengths

        quantized_query, query_scale = torch_npu.npu_dynamic_mx_quant(
            query, dst_type=torch_npu.float4_e2m1fn_x2)
        quantized_query = quantized_query.view(torch.uint8)

        packed_dim = self.width // 2
        quantized_query = quantized_query.reshape(-1, self.n_heads, packed_dim)
        query_scale = query_scale.contiguous().view(torch.uint8).reshape(
            quantized_query.shape[0], self.n_heads, self.width // 64, 2,
        )

        key, key_scale = source_cache
        key = key.contiguous()
        key_scale = key_scale.view(*key_scale.shape[:-1], 2, 2).contiguous()

        weights = weights.reshape(-1, self.n_heads).float().contiguous()

        '''print("zzx select_projected quantized_query", quantized_query.shape, quantized_query.dtype)
        print("zzx select_projected query_scale", query_scale.shape, query_scale.dtype)
        print("zzx select_projected key", key.shape, key.dtype)
        print("zzx select_projected key_scale", key_scale.shape, key_scale.dtype)
        print("zzx select_projected weights", weights.shape, weights.dtype)'''

        common = dict(
            cu_seqlens_q=source_metadata.query_start_loc,
            seqused_k=source_metadata.cache_seq_lens,
            cmp_residual_k=source_metadata.cmp_residual,
            block_table=source_metadata.block_table,
            metadata=source_metadata.qli_metadata,
            max_seqlen_q=source_metadata.max_query_len,
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_value=False,
        )
        if uses_candidate_filter:
            selected, _ = quant_sparse_lightning_indexer(
                q=quantized_query,
                k=key,
                w=weights,
                descale_q=query_scale,
                descale_k=key_scale,
                candidate_block_indices=candidate_indices,
                candidate_block_length=candidate_lengths,
                topk=topk,
                quant_mode=1, # mxpf4
                candidate_block_size=candidate_block_size,
                **common,
            )
        else:
            selected, _, cand_indices, cand_lengths = quant_lightning_indexer(
                q=quantized_query,
                k=key,
                w=weights,
                q_descale=query_scale,
                k_descale=key_scale,
                topk=topk,
                quant_mode=1, # mxpf4
                candidate_topk_blocks=candidate_topk_blocks,
                candidate_block_size=candidate_block_size,
                **common,
            )
            if is_candidate_source:
                candidate_indices = cand_indices
                candidate_lengths = cand_lengths

        selected = prepare_indexer_indices(selected.squeeze(1), positions, self.compress_ratio)
        return selected, candidate_indices, candidate_lengths
