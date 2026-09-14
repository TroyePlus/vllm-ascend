# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Engram projection and gate."""

import torch
from torch import nn
from vllm.logger import logger
from vllm.model_executor.layers.linear import ReplicatedLinear

from .engram_gate import engram_gate


class AscendEngram(nn.Module):
    """One checkpoint Engram layer with an externally prepared embedding."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.embed = None
        hash_width = (
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim
        )
        self.wkv = ReplicatedLinear(
            hash_width,
            (config.hc_mult + 1) * config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.engram.wkv",
            return_bias=False,
        )
        self.q_weight = nn.Parameter(
            torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        )
        self.k_weight = nn.Parameter(
            torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        )
        logger.info(
            "FOR-ENGRAM gate module initialized: prefix=%s hash_width=%d "
            "hidden_size=%d hc_mult=%d basis=checkpoint-native",
            prefix,
            hash_width,
            self.hidden_size,
            self.hc_mult,
        )

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if self.embed is None:
            raise RuntimeError("Engram embedding was not initialized")
        return self.embed(hash_ids).flatten(1)

    def apply_lookup(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        active_mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        kv = self.wkv(embedding)
        key, value = kv.split(
            [self.hc_mult * self.hidden_size, self.hidden_size],
            dim=-1,
        )
        output = engram_gate(
            hidden_states,
            key.view(num_tokens, self.hc_mult, self.hidden_size),
            value,
            self.q_weight.float() * self.k_weight.float(),
            active_mask,
            eps,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        active_mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return self.apply_lookup(
            hidden_states,
            self.lookup(hash_ids),
            active_mask,
            eps,
        )
