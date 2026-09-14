# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import custom_ops
import cann_ops_transformer
from safetensors import safe_open
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
    DeepseekV41EagerAttentionImpl,
)
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41FullSpec,
    DeepseekV41SWASpec,
    mxfp4_row_bytes,
    validate_cache_runtime,
)
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM,
    AscendDeepseekV4SWACache,
    DeepseekV2DecoderLayer,
    DeepseekV4Attention,
    DeepseekV4Model,
)

from .compressor import DeepseekV41Compressor, _read, text_config_of
from .engram import AscendEngram
from .engram_hash import NgramHashState
from .engram_offload import (
    ElasticEngramEmbedding,
    EngramTableState,
    create_engram_process_group,
)
from .indexer import DeepseekV41Indexer

@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""

    def __init__(self, topk_indices, candidate_indices, candidate_lengths):
        self.topk_indices = topk_indices
        self.candidate_indices = candidate_indices
        self.candidate_lengths = candidate_lengths

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    config = text_config_of(config)
    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(AscendDeepseekV4SWACache):
    """V4 execution-compatible SWA plane participating in V4.1 grouping."""

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        use_a5_quantized_cache = get_ascend_device_type() == AscendDeviceType.A5
        # DSV4's base spec may widen FP8 rows by 128 bytes. DSV4.1 owns its
        # layout, so recalculate from the configured payload dimension.
        payload_dim = self.head_dim
        head_size = (
            payload_dim + (payload_dim // 32) * torch.bfloat16.itemsize
            if use_a5_quantized_cache
            else spec.head_size
        )
        return DeepseekV41SWASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=head_size,
            dtype=torch.uint8 if use_a5_quantized_cache else spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version="deepseek_v4",
            alignment=spec.alignment,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41Attention(DeepseekV4Attention):
    """V4 projections plus V4.1 source-owned cache and fused DSA execution."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
    ):
        config = text_config_of(config)
        validate_cache_runtime(vllm_config)
        layer_idx = int(prefix.split(".")[-2])
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        # Reuse V4's quant-aware projections and stable SWA eager backend.  A
        # zero ratio prevents V4 from creating its incompatible c4/c128 planes.
        original_ratios = config.compress_ratios
        config.compress_ratios = tuple(0 for _ in original_ratios)
        try:
            super().__init__(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )
        finally:
            config.compress_ratios = original_ratios
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

        # V4.1 applies YaRN only to layers carrying long-context compressed KV.
        # Pure SWA layers use the unscaled base RoPE even though the allocated
        # lookup table still spans the configured maximum context length.
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=(config.compress_rope_theta if role.has_long_context else config.rope_theta),
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_seq_len=(max_position_embeddings if role.has_long_context else 0),
            rope_groups=["default"],
        )
        block_size = vllm_config.cache_config.block_size
        if block_size <= 0 or block_size % 2:
            raise ValueError("V4.1 logical block_size must be a positive multiple of two")
        owned = []
        if role.is_kv_source:
            owned.extend((f"{prefix}.long_kv_cache", f"{prefix}.indexer.k_cache"))
            if role.compress_ratio == 2:
                owned.append(f"{prefix}.compressor.state_cache")
        duplicates = set(owned) & vllm_config.compilation_config.static_forward_context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = _read(config, "head_dim")
        self.softmax_scale = width**-0.5
        use_a5_quantized_cache = (
            get_ascend_device_type() == AscendDeviceType.A5
        )
        if role.is_kv_source:
            # On A5 the long-context KV is stored quantized:
            # kv_compress_epilog_v2 packs mxfp4 values + bf16 scales
            # (group 16) into uint8 rows, so head_size is the packed byte
            # width rather than the element count (NPU-verified op
            # contract). Other devices keep the raw BF16 plane and the
            # builder-prepared [T, 2] scatter write.
            head_size = mxfp4_row_bytes(width) if use_a5_quantized_cache else width
            cache_dtype = torch.uint8 if use_a5_quantized_cache else torch.bfloat16
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                DeepseekV41FullSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=head_size,
                    dtype=cache_dtype,
                    compress_ratio=role.compress_ratio,
                ),
            )
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        self.v41_impl = DeepseekV41EagerAttentionImpl(
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output


class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""

    attention_cls = DeepseekV41Attention

    def __init__(self, vllm_config, prefix, **kwargs):
        super().__init__(vllm_config, prefix, **kwargs)
        config = vllm_config.model_config.hf_config
        engram_enabled = get_ascend_config().enable_engram
        quant_config = vllm_config.quant_config
        is_draft_layer = bool(kwargs.get("is_draft_layer", False))
        if engram_enabled and not is_draft_layer and self.layer_idx in config.engram_layer_ids:
            self.engram = AscendEngram(config, quant_config, prefix)
        else:
            self.engram = None
        self.ffn_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor,
               hc_scale: torch.Tensor, hc_base: torch.Tensor):
        y, post, comb, pre = torch.ops.custom.npu_hc_pre_v2(
            x=x,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            pre_mix=pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )
        return y, post, comb, pre

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)
        return y

    @staticmethod
    def hc_pre_mix(x: torch.Tensor, pre_mix: torch.Tensor):
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1)
        return y.to(x.dtype)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states, pre_mix, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states, attn_pre, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x_fp32 = x.to(torch.float32)
        x = self.mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32)
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4Model):
    """Single V4.1 backbone entry, matching ``deepseek_v4/model.py``."""

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        ascend_config = get_ascend_config()
        if ascend_config.enable_engram:
            hf_config = vllm_config.model_config.hf_config
            logger.info(
                "FOR-ENGRAM model initialization started: model=%s prefix=%s layers=%s engram_tp_size=%d storage=%s",
                vllm_config.model_config.model,
                prefix,
                getattr(hf_config, "engram_layer_ids", None),
                ascend_config.engram_tp_size,
                ascend_config.engram_storage,
            )
            if not ascend_config.enable_engram_offload:
                raise NotImplementedError("The synchronous Engram milestone requires enable_engram_offload=True")
            if vllm_config.load_config.load_format == "dummy":
                raise ValueError("Engram offload cannot be initialized from dummy weights")
            if vllm_config.load_config.safetensors_load_strategy != "lazy":
                raise ValueError("Engram offload requires --safetensors-load-strategy lazy")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # V4.1 collapses with the last block's ffn_pre; it has no hc_head
        # projection in the checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale, self.hc_norm
        topology = build_layer_plan(self.config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        candidate_length_buffer = torch.zeros(
            (max_tokens, 1),
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.candidate_length_buffer = candidate_length_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
            candidate_length_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state
        self.engram_root = vllm_config.model_config.model
        config = self.config
        self.engram_hash = None
        if ascend_config.enable_engram:
            root = Path(self.engram_root)
            if not root.is_dir():
                raise ValueError("Engram lazy loading requires a local model directory")
            with torch.device("cpu"):
                tokenizer = AutoTokenizer.from_pretrained(
                    root,
                    trust_remote_code=getattr(
                        vllm_config.model_config,
                        "trust_remote_code",
                        False,
                    ),
                )
                engram_hash = NgramHashState(config, tokenizer)
            self.engram_hash = engram_hash.to(self.topk_indices_buffer.device)
            for slot, (layer_id, rows) in enumerate(zip(config.engram_layer_ids, config.engram_num_embeddings)):
                group, owns_group = create_engram_process_group(
                    ascend_config.engram_tp_size,
                    layer_id=layer_id,
                    expected_world_size=int(
                        vllm_config.parallel_config.world_size_across_dp
                    ),
                )
                self.layers[layer_id].engram.embed = ElasticEngramEmbedding(
                    rows,
                    config.engram_head_dim,
                    ascend_config.engram_tp_size,
                    group=group,
                    owns_group=owns_group,
                    layer_id=layer_id,
                    storage_format=ascend_config.engram_storage,
                    minimum_rows=self.engram_hash.required_num_embeddings[slot],
                    device=self.topk_indices_buffer.device,
                )
        self._engram_input_buffers = None
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        if ascend_config.enable_engram:
            logger.info(
                "FOR-ENGRAM model initialization completed: layers=%s max_graph_tokens=%d device=%s",
                config.engram_layer_ids,
                self._engram_max_tokens,
                self.topk_indices_buffer.device,
            )

    def prepare_engram(
        self,
        input_ids,
        positions,
        query_start_loc=None,
        lookback_token_ids=None,
        lookback_dead_mask=None,
        current_dead_mask=None,
    ):
        """Hash and synchronously fetch embeddings outside the model graph."""
        config = self.config
        if not get_ascend_config().enable_engram:
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        if input_ids is None or self.engram_hash is None:
            raise ValueError("Engram requires raw input token IDs")
        if query_start_loc is None:
            metadata = get_forward_context().attn_metadata
            if metadata is None:
                raise ValueError("Engram requires query_start_loc for every synchronous batch")
            first = self.layers[0].self_attn.dsa_attn.swa_cache_layer
            meta = metadata[first.prefix]
            query_start_loc = (
                meta.query_start_loc_cpu
                if getattr(meta, "query_start_loc_cpu", None) is not None
                else meta.query_start_loc.detach().cpu()
            ).long()
        n = int(query_start_loc[-1])
        current_ids = input_ids[:n]
        if current_dead_mask is not None:
            current_dead_mask = current_dead_mask[:n]
        hashes, mask = self.engram_hash(
            current_ids,
            positions[:n],
            query_start_loc,
            dead_mask=current_dead_mask,
            lookback_token_ids=lookback_token_ids,
            lookback_dead_mask=lookback_dead_mask,
        )
        lookups = {}
        for slot, layer_id in enumerate(config.engram_layer_ids):
            lookups[layer_id] = self.layers[layer_id].engram.lookup(hashes[:, slot])
        return lookups, mask

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        query_start_loc=None,
        lookback_token_ids=None,
        lookback_dead_mask=None,
        current_dead_mask=None,
    ):
        """Refresh persistent inputs before main-model capture or replay."""
        engram_enabled = get_ascend_config().enable_engram
        num_requests = 0 if query_start_loc is None else query_start_loc.numel() - 1
        try:
            lookups, mask = self.prepare_engram(
                input_ids,
                positions,
                query_start_loc=query_start_loc,
                lookback_token_ids=lookback_token_ids,
                lookback_dead_mask=lookback_dead_mask,
                current_dead_mask=current_dead_mask,
            )
        except BaseException:
            if engram_enabled:
                logger.exception(
                    "FOR-ENGRAM synchronous hash/fetch failed: requests=%d",
                    num_requests,
                )
            raise
        num_tokens = positions.shape[0]
        # The compiled V4.1 backbone uses the scheduler's static token
        # capacity for decode graphs (typically max_num_batched_tokens), even
        # when the current request has one token.  Keep lookup tensors at that
        # capacity so every captured graph sees the same Engram shape.
        output_tokens = max(self._engram_max_tokens, padded_tokens or 0)
        if output_tokens < num_tokens:
            raise ValueError("Engram padded token count is smaller than the input")
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            self._engram_input_buffers = (
                {layer: values.new_zeros((capacity, values.shape[1])) for layer, values in lookups.items()},
                mask.new_zeros(capacity),
            )
        buffers, mask_buffer = self._engram_input_buffers
        padded_mask = mask_buffer[:output_tokens]
        padded_mask.zero_()
        padded_mask[: mask.numel()].copy_(mask)
        padded_lookups = {}
        for layer, values in lookups.items():
            padded = buffers[layer][:output_tokens]
            padded.zero_()
            padded[: values.shape[0]].copy_(values)
            padded_lookups[layer] = padded
        return {"engram_lookups": padded_lookups, "engram_mask": padded_mask}

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("V4.1 eager milestone currently requires PP=1")
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            lookups, token_mask = self.prepare_engram(input_ids, positions)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        self.shared_attention_state.reset()
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        aux_hidden_states = []
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers:
            last_layer = layer
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))
            if layer.engram is not None and token_mask.numel():
                n = hidden_states.shape[0]
                # Graph captures keep lookup buffers at static capacity; the
                # model's actual token dimension remains scheduler-dynamic.
                lookup = lookups[layer.layer_idx][:n]
                active_mask = token_mask[:n]
                hidden_states[:n] = layer.engram.apply_lookup(
                    hidden_states[:n],
                    lookup,
                    active_mask,
                    self.config.rms_norm_eps,
                )
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=moe_input_ids)
        assert last_layer is not None
        hidden_states = last_layer.hc_pre_mix(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def offload_weights(self, checkpoint_keys: dict[int, str]):
        if self.engram_hash is None:
            return
        logger.info(
            "FOR-ENGRAM model offload started: layers=%s",
            self.config.engram_layer_ids,
        )
        try:
            for layer_id in self.config.engram_layer_ids:
                embedding = self.layers[layer_id].engram.embed
                if embedding.state is EngramTableState.READY:
                    continue
                if embedding.state is EngramTableState.EMPTY:
                    try:
                        checkpoint_key = checkpoint_keys[layer_id]
                    except KeyError as exc:
                        raise ValueError(f"Missing Engram checkpoint key for layer {layer_id}") from exc
                    embedding.load_checkpoint(self.engram_root, checkpoint_key)
                embedding.offload_weights()
        except BaseException:
            logger.exception("FOR-ENGRAM model offload failed; destroying initialized resources")
            try:
                self.destroy_engram()
            except BaseException:
                logger.exception(
                    "FOR-ENGRAM model offload cleanup also failed; preserving "
                    "the original offload error"
                )
            raise
        logger.info(
            "FOR-ENGRAM model offload completed: layers=%s",
            self.config.engram_layer_ids,
        )

    def destroy_engram(self):
        if self.engram_hash is None:
            return
        logger.debug(
            "FOR-ENGRAM model destroy started: layers=%s",
            self.config.engram_layer_ids,
        )
        first_error = None
        for layer_id in reversed(self.config.engram_layer_ids):
            try:
                self.layers[layer_id].engram.embed.destroy()
            except BaseException as exc:
                logger.exception(
                    "FOR-ENGRAM table destroy failed during model cleanup: layer=%d",
                    layer_id,
                )
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            logger.error("FOR-ENGRAM model destroy completed with errors")
            raise RuntimeError(
                "One or more Engram tables failed to destroy"
            ) from first_error
        logger.debug("FOR-ENGRAM model destroy completed")


class AscendDeepseekV41ForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = DeepseekV41Model
    requires_raw_input_tokens = True
    _DEFERRED_WEIGHT_MARKERS = ()
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        query_start_loc=None,
        lookback_token_ids=None,
        lookback_dead_mask=None,
        current_dead_mask=None,
    ):
        return self.model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
            query_start_loc=query_start_loc,
            lookback_token_ids=lookback_token_ids,
            lookback_dead_mask=lookback_dead_mask,
            current_dead_mask=current_dead_mask,
        )

    def offload_weights(self):
        checkpoint_keys = getattr(self, "_engram_checkpoint_keys", None)
        if checkpoint_keys is None:
            raise RuntimeError("Engram checkpoint keys were not registered before offload")
        self.model.offload_weights(checkpoint_keys)

    def destroy_engram(self):
        self.model.destroy_engram()

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
        )

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not get_ascend_config().enable_engram:
            return super().load_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        logger.info(
            "FOR-ENGRAM checkpoint weight loading started: expected_layers=%s",
            self.model.config.engram_layer_ids,
        )
        checkpoint_keys: dict[int, str] = {}

        def milestone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.endswith(".engram.embed.scale"):
                    # Consumed together with the FP8 table by load_checkpoint.
                    continue
                if name.endswith(".engram.embed.weight"):
                    parts = name.split(".")
                    try:
                        layer_id = int(parts[parts.index("layers") + 1])
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f"Cannot determine Engram layer from {name}") from exc
                    if layer_id in checkpoint_keys:
                        raise ValueError(f"Duplicate Engram embedding table for layer {layer_id}")
                    checkpoint_keys[layer_id] = name
                    continue
                if self._is_milestone_weight(name):
                    yield name, tensor

        try:
            loaded = super().load_weights(milestone_weights())
            expected_tables = set(self.model.config.engram_layer_ids)
            loaded_tables = set(checkpoint_keys)
            if loaded_tables != expected_tables:
                logger.error(
                    "FOR-ENGRAM checkpoint weight loading incomplete: missing_layers=%s unexpected_layers=%s",
                    expected_tables - loaded_tables,
                    loaded_tables - expected_tables,
                )
                raise ValueError(
                    "Engram embedding table mismatch: "
                    f"missing={expected_tables - loaded_tables}, "
                    f"unexpected={loaded_tables - expected_tables}"
                )
            self._engram_checkpoint_keys = checkpoint_keys
        except BaseException:
            logger.exception("FOR-ENGRAM checkpoint loading failed; destroying table resources")
            try:
                self.model.destroy_engram()
            except BaseException:
                logger.exception("FOR-ENGRAM checkpoint failure cleanup also failed")
            raise
        logger.info(
            "FOR-ENGRAM checkpoint weight loading completed: layers=%s",
            sorted(loaded_tables),
        )
        return loaded
