# SPDX-License-Identifier: Apache-2.0

import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import vllm_ascend.models.deepseek_v41.engram_offload as engram_offload

from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v41.engram_gate import engram_gate
from vllm_ascend.models.deepseek_v41.engram_hash import (
    EngramTokenSequence,
    NgramHashState,
    build_lookback_token_ids,
)
from vllm_ascend.models.deepseek_v41.engram_offload import (
    ElasticEngramEmbedding,
    EngramTableState,
    create_engram_process_group,
)
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForConditionalGeneration,
)


class _BackendTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return self.tokens[token_ids[0]]

    def id_to_token(self, token_id):
        return self.tokens[token_id]


class _Tokenizer:
    def __init__(self):
        self.tokens = [
            "<pad>",
            "alpha",
            "beta",
            "gamma",
            "<image>",
            "<image-pad>",
            "delta",
        ]
        self.backend_tokenizer = _BackendTokenizer(self.tokens)

    def __len__(self):
        return len(self.tokens)


def _make_hash_state():
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[112, 252],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_vocab_size=11,
        engram_pad_id=0,
        engram_compressed_vocab_size=7,
        image_token_id=4,
        image_pad_token_id=5,
    )
    return NgramHashState(config, _Tokenizer())


def test_build_lookback_is_newest_first_and_marks_image_history_dead():
    token_ids = np.array(
        [
            [10, 11, 4, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=np.int32,
    )

    lookback, dead = build_lookback_token_ids(
        token_ids,
        np.array([4, 0], dtype=np.int32),
        depth=3,
        image_token_id=4,
        image_pad_token_id=5,
    )

    np.testing.assert_array_equal(lookback, [[13, 4, 11], [-1, -1, -1]])
    np.testing.assert_array_equal(dead, [[False, True, False], [True, True, True]])


def test_build_lookback_fails_when_authoritative_history_is_missing():
    histories = [EngramTokenSequence([10, 11], [12])]

    with pytest.raises(ValueError, match="missing authoritative text token history"):
        build_lookback_token_ids(
            histories,
            np.array([4], dtype=np.int32),
            depth=3,
            image_token_id=4,
            image_pad_token_id=5,
        )


def test_build_lookback_marks_explicit_image_spans_dead():
    histories = [EngramTokenSequence([10, 11, 12, 13], [])]

    lookback, dead = build_lookback_token_ids(
        histories,
        np.array([4], dtype=np.int32),
        depth=3,
        image_token_id=4,
        image_pad_token_id=5,
        image_spans=[[(1, 2)]],
    )

    np.testing.assert_array_equal(lookback, [[13, 12, 11]])
    np.testing.assert_array_equal(dead, [[False, True, True]])


def test_chunk_hash_matches_same_positions_in_full_prefill():
    state = _make_hash_state()
    all_ids = torch.tensor([1, 2, 3, 6, 1, 2])
    full_lookback = torch.full((1, 3), -1)
    full_dead = torch.ones((1, 3), dtype=torch.bool)
    full_hash, full_mask = state(
        all_ids,
        torch.arange(6),
        torch.tensor([0, 6]),
        lookback_token_ids=full_lookback,
        lookback_dead_mask=full_dead,
    )

    chunk_ids = all_ids[3:]
    chunk_lookback = torch.tensor([[3, 2, 1]])
    chunk_hash, chunk_mask = state(
        chunk_ids,
        torch.arange(3, 6),
        torch.tensor([0, 3]),
        lookback_token_ids=chunk_lookback,
        lookback_dead_mask=torch.zeros_like(chunk_lookback, dtype=torch.bool),
    )

    torch.testing.assert_close(chunk_hash, full_hash[3:])
    torch.testing.assert_close(chunk_mask, full_mask[3:])


def test_image_token_is_a_barrier_and_disables_its_own_gate():
    state = _make_hash_state()
    first_hash, first_mask = state(
        torch.tensor([1, 2, 4, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
    )
    second_hash, second_mask = state(
        torch.tensor([6, 1, 4, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
    )

    assert not first_mask[2]
    assert not second_mask[2]
    torch.testing.assert_close(first_hash[3], second_hash[3])


def test_engram_gate_matches_recipes_checkpoint_native_formula():
    torch.manual_seed(0)
    hidden = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    key = torch.randn_like(hidden)
    value = torch.randn(3, 8, dtype=torch.bfloat16)
    channel_weight = torch.randn(2, 8, dtype=torch.float32)
    token_mask = torch.tensor([True, False, True])
    eps = 1e-6

    hidden_float = hidden.float()
    key_float = key.float()
    rstd = torch.rsqrt(hidden_float.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key_float.square().mean(-1) + eps)
    dot = (
        (hidden_float * channel_weight * key_float).sum(-1)
        * rstd
        * hidden.shape[-1] ** -0.5
    )
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, magnitude, -magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    expected = (
        hidden_float + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
    ).to(hidden.dtype)

    actual = engram_gate(
        hidden,
        key,
        value,
        channel_weight,
        token_mask,
        eps,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], hidden[1], rtol=0, atol=0)


def test_current_placeholder_mask_is_a_hash_barrier_after_sanitization():
    state = _make_hash_state()
    dead_mask = torch.tensor([False, True, False, False])
    first_hash, first_mask = state(
        torch.tensor([1, 0, 2, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
        dead_mask=dead_mask,
    )
    second_hash, second_mask = state(
        torch.tensor([6, 0, 2, 3]),
        torch.arange(4),
        torch.tensor([0, 4]),
        dead_mask=dead_mask,
    )

    assert not first_mask[1]
    assert not second_mask[1]
    torch.testing.assert_close(first_hash[2:], second_hash[2:])


def test_tp1_creates_a_dedicated_non_null_process_group():
    process_group = object()
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=0,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_backend",
            return_value="hccl",
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group",
            return_value=process_group,
        ) as new_group,
    ):
        group, owns_group = create_engram_process_group(
            1,
            expected_world_size=1,
        )

    assert group is process_group
    assert owns_group
    new_group.assert_called_once_with(ranks=[0], backend="hccl")


def test_process_group_rejects_runtime_world_mismatch_before_creation():
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=4,
        ),
        patch("vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group") as new_group,
        pytest.raises(RuntimeError, match="runtime=4, configured=8"),
    ):
        create_engram_process_group(4, expected_world_size=8)

    new_group.assert_not_called()


def test_process_group_selects_contiguous_group_from_global_world():
    first_group = object()
    second_group = object()
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_world_size",
            return_value=8,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=5,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_backend",
            return_value="hccl",
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.new_group",
            side_effect=[first_group, second_group],
        ) as new_group,
    ):
        group, owns_group = create_engram_process_group(
            4,
            expected_world_size=8,
        )

    assert group is second_group
    assert owns_group
    assert [entry.kwargs["ranks"] for entry in new_group.call_args_list] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_vl_wrapper_forwards_engram_inputs_and_lifecycle():
    language_model = MagicMock()
    wrapper = SimpleNamespace(language_model=language_model)
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    query_start_loc = torch.tensor([0, 2])
    lookback_token_ids = torch.tensor([[-1, -1, -1]])
    lookback_dead_mask = torch.ones_like(lookback_token_ids, dtype=torch.bool)
    current_dead_mask = torch.tensor([False, True])

    AscendDeepseekV41ForConditionalGeneration.prepare_engram_inputs(
        wrapper,
        input_ids,
        positions,
        8,
        query_start_loc=query_start_loc,
        lookback_token_ids=lookback_token_ids,
        lookback_dead_mask=lookback_dead_mask,
        current_dead_mask=current_dead_mask,
    )
    AscendDeepseekV41ForConditionalGeneration.offload_weights(wrapper)
    AscendDeepseekV41ForConditionalGeneration.destroy_engram(wrapper)

    language_model.prepare_engram_inputs.assert_called_once_with(
        input_ids,
        positions,
        8,
        query_start_loc=query_start_loc,
        lookback_token_ids=lookback_token_ids,
        lookback_dead_mask=lookback_dead_mask,
        current_dead_mask=current_dead_mask,
    )
    language_model.offload_weights.assert_called_once_with()
    language_model.destroy_engram.assert_called_once_with()


def test_engram_checkpoint_collectives_are_deferred_until_post_load():
    wrapper = AscendDeepseekV41ForCausalLM.__new__(AscendDeepseekV41ForCausalLM)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        destroy_engram=MagicMock(),
    )
    object.__setattr__(wrapper, "model", model)
    consumed = []

    def load_base(_self, weights):
        consumed.extend(name for name, _ in weights)
        return set(consumed)

    weights = [
        ("model.layers.14.engram.embed.weight", torch.empty(0)),
        ("model.layers.0.self_attn.weight", torch.empty(0)),
        ("model.layers.1.engram.embed.scale", torch.empty(0)),
        ("model.layers.1.engram.embed.weight", torch.empty(0)),
        ("model.layers.14.engram.embed.scale", torch.empty(0)),
    ]
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.model.get_ascend_config",
            return_value=SimpleNamespace(enable_engram=True),
        ),
        patch.object(
            AscendDeepseekV4ForCausalLM,
            "load_weights",
            new=load_base,
        ),
    ):
        loaded = wrapper.load_weights(weights)

    assert consumed == ["model.layers.0.self_attn.weight"]
    assert loaded == {"model.layers.0.self_attn.weight"}
    assert wrapper._engram_checkpoint_keys == {
        1: "model.layers.1.engram.embed.weight",
        14: "model.layers.14.engram.embed.weight",
    }
    model.destroy_engram.assert_not_called()


def test_engram_post_load_offload_uses_configured_layer_order():
    events = []

    class _Embedding:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.state = EngramTableState.EMPTY

        def load_checkpoint(self, root, key):
            events.append(("load", self.layer_id, root, key))
            self.state = EngramTableState.STAGED

        def offload_weights(self):
            events.append(("offload", self.layer_id))
            self.state = EngramTableState.READY

    layers = [SimpleNamespace(engram=None) for _ in range(15)]
    layers[1].engram = SimpleNamespace(embed=_Embedding(1))
    layers[14].engram = SimpleNamespace(embed=_Embedding(14))
    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        layers=layers,
        engram_root="checkpoint",
    )

    DeepseekV41Model.offload_weights(
        model,
        {
            14: "model.layers.14.engram.embed.weight",
            1: "model.layers.1.engram.embed.weight",
        },
    )
    DeepseekV41Model.offload_weights(model, {})

    assert events == [
        ("load", 1, "checkpoint", "model.layers.1.engram.embed.weight"),
        ("offload", 1),
        ("load", 14, "checkpoint", "model.layers.14.engram.embed.weight"),
        ("offload", 14),
    ]


def test_model_destroy_attempts_every_engram_table():
    events = []

    class _Embedding:
        def __init__(self, layer_id, fails=False):
            self.layer_id = layer_id
            self.fails = fails

        def destroy(self):
            events.append(self.layer_id)
            if self.fails:
                raise ValueError(f"destroy {self.layer_id}")

    layers = [SimpleNamespace(engram=None) for _ in range(15)]
    layers[1].engram = SimpleNamespace(embed=_Embedding(1))
    layers[14].engram = SimpleNamespace(embed=_Embedding(14, fails=True))
    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1, 14]),
        layers=layers,
    )

    with pytest.raises(RuntimeError, match="failed to destroy") as exc_info:
        DeepseekV41Model.destroy_engram(model)

    assert events == [14, 1]
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_offload_preserves_primary_error_when_cleanup_fails():
    class _Embedding:
        state = EngramTableState.EMPTY

        def load_checkpoint(self, root, key):
            del root, key
            raise ValueError("primary load failure")

        def offload_weights(self):
            raise AssertionError("offload must not run after load failure")

        def destroy(self):
            raise RuntimeError("cleanup failure")

    model = SimpleNamespace(
        engram_hash=object(),
        config=SimpleNamespace(engram_layer_ids=[1]),
        layers=[SimpleNamespace(engram=None), SimpleNamespace(engram=SimpleNamespace(embed=_Embedding()))],
        engram_root="checkpoint",
    )
    model.destroy_engram = MethodType(DeepseekV41Model.destroy_engram, model)

    with pytest.raises(ValueError, match="primary load failure"):
        DeepseekV41Model.offload_weights(
            model,
            {1: "model.layers.1.engram.embed.weight"},
        )


class _TensorSlice:
    def __init__(self, tensor, dtype):
        self.tensor = tensor
        self.dtype = dtype

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return self.dtype

    def __getitem__(self, item):
        return self.tensor[item]


class _SafeFile:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get_slice(self, key):
        return self.tensors[key]


def test_fp8_loader_only_decodes_local_rows_and_zero_pads_tail(tmp_path):
    weight_key = "model.layers.1.engram.embed.weight"
    scale_key = "model.layers.1.engram.embed.scale"
    weight = torch.arange(20, dtype=torch.float32).view(5, 4)
    scale = torch.full((5, 2), 2.0)
    tensors = {
        weight_key: _TensorSlice(weight, "F8_E4M3FN"),
        scale_key: _TensorSlice(scale, "F8_E8M0"),
    }
    table = None
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=1,
        ),
        patch.object(
            ElasticEngramEmbedding,
            "_weight_map",
            return_value={weight_key: "table.safetensors", scale_key: "table.safetensors"},
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.safe_open",
            return_value=_SafeFile(tensors),
        ),
        patch.object(
            ElasticEngramEmbedding,
            "_sync_phase_status",
            return_value=True,
        ),
    ):
        table = ElasticEngramEmbedding(5, 4, 2, group=object())
        table.load_checkpoint(tmp_path, weight_key)

    assert table is not None
    assert table.state is EngramTableState.STAGED
    expected = (weight[3:5] * 2).to(torch.bfloat16)
    torch.testing.assert_close(table._host_weight[:2], expected)
    torch.testing.assert_close(table._host_weight[2], torch.zeros(4, dtype=torch.bfloat16))


def test_uint8_e8m0_loader_decodes_local_rows_to_bf16(tmp_path):
    weight_key = "model.layers.1.engram.embed.weight"
    scale_key = "model.layers.1.engram.embed.scale"
    weight = torch.ones((5, 32), dtype=torch.float32)
    scale_bits = torch.tensor([[127], [127], [127], [126], [128]], dtype=torch.uint8)
    tensors = {
        weight_key: _TensorSlice(weight, "F8_E4M3FN"),
        scale_key: _TensorSlice(scale_bits, "U8"),
    }
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=1,
        ),
        patch.object(
            ElasticEngramEmbedding,
            "_weight_map",
            return_value={weight_key: "table.safetensors", scale_key: "table.safetensors"},
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.safe_open",
            return_value=_SafeFile(tensors),
        ),
        patch.object(
            ElasticEngramEmbedding,
            "_sync_phase_status",
            return_value=True,
        ),
    ):
        table = ElasticEngramEmbedding(
            6,
            32,
            2,
            group=object(),
            minimum_rows=5,
        )
        table.load_checkpoint(tmp_path, weight_key)

    expected = torch.stack(
        [
            torch.full((32,), 0.5, dtype=torch.bfloat16),
            torch.full((32,), 2.0, dtype=torch.bfloat16),
            torch.zeros(32, dtype=torch.bfloat16),
        ]
    )
    assert table.state is EngramTableState.STAGED
    assert table._host_weight.dtype == torch.bfloat16
    torch.testing.assert_close(table._host_weight, expected)


def test_checkpoint_metadata_rejects_empty_scale_columns():
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(5, 4, 1, group=object())

    with pytest.raises(ValueError, match="incompatible scale shape"):
        table._validate_checkpoint_metadata(
            "weight",
            [5, 4],
            "F8_E4M3FN",
            "scale",
            [5, 0],
            "F8_E8M0",
        )


def test_checkpoint_metadata_rejects_rows_below_hash_coverage():
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(
            6,
            32,
            1,
            group=object(),
            minimum_rows=5,
        )

    with pytest.raises(ValueError, match="checkpoint rows must be in"):
        table._validate_checkpoint_metadata(
            "weight",
            [4, 32],
            "F8_E4M3FN",
            "scale",
            [4, 1],
            "F8_E8M0",
        )


def test_elastic_fetch_waits_once_and_destroy_is_idempotent():
    class _Buffer:
        def __init__(self):
            self.fetches = 0
            self.waits = 0
            self.destroys = 0

        def engram_fetch(self, ids):
            self.fetches += 1

            def wait():
                self.waits += 1
                return torch.zeros((ids.numel(), 4), dtype=torch.bfloat16)

            return wait

        def destroy(self):
            self.destroys += 1

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._state = EngramTableState.READY

    output = table(torch.tensor([[0, 1], [2, 3]], dtype=torch.int64))
    table.destroy()
    table.destroy()

    assert output.shape == (2, 2, 4)
    assert buffer.fetches == 1
    assert buffer.waits == 1
    assert buffer.destroys == 1


def test_mixed_fetch_abi_error_switches_to_legacy_aclnn_bridge():
    class _Buffer:
        def __init__(self):
            self._engram_fetch_in_progress = False

        def engram_fetch(self, ids):
            del ids
            self._engram_fetch_in_progress = True
            raise RuntimeError(
                "call aclnnEngramFetch failed: Parameter fetched of EngramFetch "
                "has incorrect shape dim 1D; fetched must be 2D"
            )

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._state = EngramTableState.READY
    expected = torch.zeros((2, 4), dtype=torch.bfloat16)

    with (
        patch.object(table, "_fetch_legacy_aclnn", return_value=expected) as legacy_fetch,
        patch.object(engram_offload, "_ELASTIC_BUFFER_FETCH_ABI", "auto"),
    ):
        actual = table(torch.tensor([0, 1], dtype=torch.int32))
        assert engram_offload._ELASTIC_BUFFER_FETCH_ABI == "legacy-aclnn"

    torch.testing.assert_close(actual, expected)
    assert not buffer._engram_fetch_in_progress
    legacy_fetch.assert_called_once()


def test_unknown_fetch_error_is_not_hidden_by_legacy_bridge():
    class _Buffer:
        def engram_fetch(self, ids):
            del ids
            raise RuntimeError("transport failed")

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    table.engram_buffer = _Buffer()
    table._state = EngramTableState.READY

    with (
        patch.object(table, "_fetch_legacy_aclnn") as legacy_fetch,
        patch.object(engram_offload, "_ELASTIC_BUFFER_FETCH_ABI", "auto"),
        pytest.raises(RuntimeError, match="transport failed"),
    ):
        table(torch.tensor([0], dtype=torch.int32))

    legacy_fetch.assert_not_called()
    assert table.state is EngramTableState.POISONED


def test_cached_legacy_fetch_abi_skips_public_interface():
    class _Buffer:
        def engram_fetch(self, ids):
            del ids
            raise AssertionError("public interface must not be called")

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    table.engram_buffer = _Buffer()
    table._state = EngramTableState.READY
    expected = torch.zeros((1, 4), dtype=torch.bfloat16)

    with (
        patch.object(table, "_fetch_legacy_aclnn", return_value=expected) as legacy_fetch,
        patch.object(engram_offload, "_ELASTIC_BUFFER_FETCH_ABI", "legacy-aclnn"),
    ):
        actual = table(torch.tensor([0], dtype=torch.int32))

    torch.testing.assert_close(actual, expected)
    legacy_fetch.assert_called_once()


def test_legacy_aclnn_bridge_receives_initialized_fetch_attributes():
    context = torch.tensor([1], dtype=torch.int32)

    class _Buffer:
        _engram_context_tensor = context
        _engram_num_entries = 8

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(8, 4, 1, group=object())
    table.engram_buffer = _Buffer()
    ids = torch.tensor([0, 1], dtype=torch.int32)

    def legacy_fetch(actual_context, actual_ids, hidden_size, num_entries):
        assert actual_context is context
        assert actual_ids is ids
        assert hidden_size == 4
        assert num_entries == 8
        return torch.ones((2, 4), dtype=torch.bfloat16)

    def legacy_wait(actual_context, fetched):
        assert actual_context is context
        return fetched

    with (
        patch.object(
            engram_offload,
            "_load_legacy_engram_fetch_op",
            return_value=SimpleNamespace(engram_fetch=legacy_fetch),
        ),
        patch.object(
            torch.ops.cann_ops_transformer,
            "engram_fetch_wait",
            side_effect=legacy_wait,
            create=True,
        ),
    ):
        actual = table._fetch_legacy_aclnn(ids)

    torch.testing.assert_close(actual, torch.ones((2, 4), dtype=torch.bfloat16))


def test_legacy_bridge_loader_recovers_stale_torch_extension_lock(tmp_path):
    loaded_module = SimpleNamespace(engram_fetch=object())
    load_calls = []
    torch_lock = tmp_path / "lock"
    torch_lock.write_text("stale", encoding="utf-8")

    class _OpBuilder:
        def __init__(self, name):
            self.name = name

        def load(self, verbose=True):
            load_calls.append(verbose)
            assert not torch_lock.exists()
            return loaded_module

    package = ModuleType("cann_ops_transformer")
    op_builder = ModuleType("cann_ops_transformer.op_builder")
    op_builder.OpBuilder = _OpBuilder
    package.op_builder = op_builder

    with (
        patch.dict(
            sys.modules,
            {
                "cann_ops_transformer": package,
                "cann_ops_transformer.op_builder": op_builder,
            },
        ),
        patch(
            "torch.utils.cpp_extension._get_build_directory",
            return_value=str(tmp_path),
        ),
        patch.object(engram_offload, "_LEGACY_ENGRAM_FETCH_OP", None),
    ):
        first = engram_offload._load_legacy_engram_fetch_op()
        second = engram_offload._load_legacy_engram_fetch_op()

    assert first is loaded_module
    assert second is loaded_module
    assert load_calls == [False]
    assert not torch_lock.exists()
    assert (tmp_path / ".vllm_ascend_build.lock").is_file()


def test_offload_writes_bf16_storage_and_releases_staging_copy():
    class _ElasticBuffer:
        instance = None

        @staticmethod
        def get_engram_storage_size_hint(rows, width, dtype):
            assert (rows, width, dtype) == (2, 4, torch.bfloat16)
            return 4096

        def __init__(self, group, **kwargs):
            assert group is process_group
            assert kwargs == {
                "num_cpu_bytes": 4096,
                "explicitly_destroy": True,
            }
            self.storage = None
            _ElasticBuffer.instance = self

        def engram_write(self, storage):
            self.storage = storage.clone()

        def destroy(self):
            return None

    package = ModuleType("cann_ops_transformer")
    ops = ModuleType("cann_ops_transformer.ops")
    ops.ElasticBuffer = _ElasticBuffer
    package.ops = ops
    process_group = object()
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(2, 4, 1, group=process_group)
    table._host_weight = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    table._state = EngramTableState.STAGED

    with patch.dict(
        sys.modules,
        {
            "cann_ops_transformer": package,
            "cann_ops_transformer.ops": ops,
        },
    ):
        table.offload_weights()

    assert table.state is EngramTableState.READY
    assert table._host_weight is None
    assert _ElasticBuffer.instance is not None
    assert _ElasticBuffer.instance.storage.dtype == torch.bfloat16


def test_offload_write_failure_poisons_table_and_releases_staging():
    class _ElasticBuffer:
        instance = None

        @staticmethod
        def get_engram_storage_size_hint(rows, width, dtype):
            return 4096

        def __init__(self, group, **kwargs):
            self.destroys = 0
            _ElasticBuffer.instance = self

        def engram_write(self, storage):
            raise RuntimeError("write failed")

        def destroy(self):
            self.destroys += 1

    package = ModuleType("cann_ops_transformer")
    ops = ModuleType("cann_ops_transformer.ops")
    ops.ElasticBuffer = _ElasticBuffer
    package.ops = ops
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(2, 4, 1, group=object())
    table._host_weight = torch.zeros((2, 4), dtype=torch.bfloat16)
    table._state = EngramTableState.STAGED

    with (
        patch.dict(
            sys.modules,
            {
                "cann_ops_transformer": package,
                "cann_ops_transformer.ops": ops,
            },
        ),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        table.offload_weights()

    assert table.state is EngramTableState.POISONED
    assert table._host_weight is None
    assert _ElasticBuffer.instance.destroys == 1


def test_native_mxfp8_storage_is_rejected_for_current_elastic_buffer_abi():
    with (
        patch(
            "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
            return_value=0,
        ),
        pytest.raises(ValueError, match="supports only bf16 storage"),
    ):
        ElasticEngramEmbedding(
            2,
            32,
            1,
            group=object(),
            storage_format="mxfp8",
        )


def test_failed_wait_poisons_table_until_destroy():
    class _Buffer:
        def __init__(self):
            self.destroys = 0

        def engram_fetch(self, ids):
            del ids

            def wait():
                raise RuntimeError("wait failed")

            return wait

        def destroy(self):
            self.destroys += 1

    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(2, 4, 1, group=object())
    buffer = _Buffer()
    table.engram_buffer = buffer
    table._state = EngramTableState.READY

    with pytest.raises(RuntimeError, match="wait failed"):
        table(torch.tensor([0]))
    assert table.state is EngramTableState.POISONED
    with pytest.raises(RuntimeError, match="not ready"):
        table(torch.tensor([0]))

    table.destroy()
    assert table.state is EngramTableState.DESTROYED
    assert buffer.destroys == 1


def test_distributed_preflight_reports_remote_failure():
    def fail_one_rank(status, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is process_group
        status.zero_()

    process_group = object()
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.get_rank",
        return_value=0,
    ):
        table = ElasticEngramEmbedding(
            4,
            4,
            2,
            group=process_group,
            device="cpu",
        )
    with patch(
        "vllm_ascend.models.deepseek_v41.engram_offload.dist.all_reduce",
        side_effect=fail_one_rank,
    ):
        assert not table._sync_phase_status(True, "test")
