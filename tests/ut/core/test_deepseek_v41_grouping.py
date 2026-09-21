# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache-group shaping: diversion merges the SWA scheduler groups.

Pool-OFF keeps the slot-aligned 4-per-group split (each group's four layers
map one-to-one onto the four shared slot tensors). Pool-ON diverts all 40
SWA layers into per-layer ring tensors that share one allocation per
request, so a single group — and a single identical ring block table —
must suffice.
"""

import torch

from vllm_ascend.core.deepseek_v41 import (
    STATE_RING_ROWS,
    DeepseekV41CompressorStateSpec,
    DeepseekV41DraftSWASpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
    group_cache_specs,
)


def _v41_specs(with_draft: bool) -> dict:
    specs = {}
    for slot_idx, layer in enumerate((2, 8, 14, 20)):
        ratio = 2 if slot_idx < 3 else 1
        specs[f"model.layers.{layer}.self_attn.long_kv_cache"] = DeepseekV41FullSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            compress_ratio=ratio,
        )
        specs[f"model.layers.{layer}.indexer.k_cache"] = DeepseekV41IndexerSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=288,
            dtype=torch.uint8,
            scale_dim=16,
            scale_dtype=torch.bfloat16,
            compress_ratio=ratio,
        )
    for layer in (2, 8, 14):
        specs[f"model.layers.{layer}.self_attn.state_ring"] = DeepseekV41CompressorStateSpec(
            block_size=STATE_RING_ROWS,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.float32,
        )
    for layer in range(40):
        specs[f"model.layers.{layer}.self_attn.swa_cache"] = DeepseekV41SWASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            sliding_window=128,
        )
    if with_draft:
        for mtp in range(3):
            specs[f"model.mtp.{mtp}.self_attn.swa_cache"] = DeepseekV41DraftSWASpec(
                block_size=128,
                num_kv_heads=1,
                head_size=192,
                dtype=torch.bfloat16,
                sliding_window=128,
            )
    return specs


def _swa_groups(groups):
    return [
        g
        for g in groups
        if g.kv_cache_specs
        and all(isinstance(s, DeepseekV41SWASpec) for s in g.kv_cache_specs.values())
    ]


def test_diverted_swa_layers_form_one_group():
    groups = group_cache_specs(_v41_specs(with_draft=True), divert_swa=True)
    assert len(groups) == 4
    swa_groups = _swa_groups(groups)
    assert len(swa_groups) == 1
    merged = swa_groups[0]
    assert len(merged.kv_cache_specs) == 40
    assert {name.rsplit(".", 1)[0] for name in merged.kv_cache_specs} == {
        f"model.layers.{layer}.self_attn" for layer in range(40)
    }
    draft_groups = [
        g
        for g in groups
        if g.kv_cache_specs
        and all(
            isinstance(s, DeepseekV41DraftSWASpec)
            for s in g.kv_cache_specs.values()
        )
    ]
    assert len(draft_groups) == 1
    assert len(draft_groups[0].kv_cache_specs) == 3


def test_diverted_groups_without_draft_stay_three():
    groups = group_cache_specs(_v41_specs(with_draft=False), divert_swa=True)
    assert len(groups) == 3
    assert len(_swa_groups(groups)) == 1


def test_shared_swa_layers_keep_slot_aligned_groups():
    groups = group_cache_specs(_v41_specs(with_draft=False), divert_swa=False)
    assert len(groups) == 12
    swa_groups = _swa_groups(groups)
    assert len(swa_groups) == 10
    assert all(len(g.kv_cache_specs) == 4 for g in swa_groups)
    covered = [name for g in swa_groups for name in g.kv_cache_specs]
    assert len(covered) == 40 == len(set(covered))
