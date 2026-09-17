# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the private-circle mixed-batch positional split."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41EagerAttentionImpl,
    _assert_decode_rows_first,
    _validate_batch_layout,
)


def _common(prefill_flags):
    return SimpleNamespace(
        is_prefilling=torch.tensor(prefill_flags, dtype=torch.bool),
    )


def test_guard_accepts_decode_rows_before_prefill_rows():
    _assert_decode_rows_first(_common([False, False, True]), num_reqs=3, num_decodes=2)


def test_guard_accepts_pure_decode_batch():
    _assert_decode_rows_first(_common([False, False]), num_reqs=2, num_decodes=2)


def test_guard_accepts_pure_prefill_batch():
    _assert_decode_rows_first(_common([True, True]), num_reqs=2, num_decodes=0)


def test_guard_rejects_prefill_row_before_decode_rows():
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([True, False]), num_reqs=2, num_decodes=1)


def test_guard_rejects_interleaved_prefill_row():
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([False, True, False]), num_reqs=3, num_decodes=2)


def test_guard_skips_when_is_prefilling_unavailable():
    _assert_decode_rows_first(SimpleNamespace(), num_reqs=2, num_decodes=1)
    _assert_decode_rows_first(
        SimpleNamespace(is_prefilling=None), num_reqs=2, num_decodes=1
    )


def test_guard_rejects_only_rows_below_num_decodes():
    # A prefill row at or after num_decodes is the expected layout.
    _assert_decode_rows_first(_common([False, True, True]), num_reqs=3, num_decodes=1)
    with pytest.raises(RuntimeError, match="decode rows before prefill"):
        _assert_decode_rows_first(_common([True, False, True]), num_reqs=3, num_decodes=1)


def _batch_layout_common(num_reqs=2, num_tokens=2):
    return SimpleNamespace(
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.full((num_tokens,), -1, dtype=torch.int32),
    )


def test_batch_layout_accepts_graph_padding():
    _validate_batch_layout(
        _batch_layout_common(num_reqs=4, num_tokens=4),
        num_reqs=4,
        num_actual_reqs=2,
        num_input_tokens=4,
        num_actual_tokens=2,
    )


@pytest.mark.parametrize(
    ("num_reqs", "num_actual_reqs", "num_input_tokens", "num_actual_tokens"),
    [
        (2, 3, 2, 2),
        (2, 2, 2, 3),
    ],
)
def test_batch_layout_rejects_actual_counts_above_padded_shape(
    num_reqs, num_actual_reqs, num_input_tokens, num_actual_tokens
):
    with pytest.raises(ValueError):
        _validate_batch_layout(
            _batch_layout_common(num_reqs=num_reqs, num_tokens=num_input_tokens),
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_input_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
        )


def test_batch_layout_requires_all_padded_request_rows():
    common = _batch_layout_common(num_reqs=2)
    common.block_table_tensor = common.block_table_tensor[:1]
    with pytest.raises(ValueError, match="block table row"):
        _validate_batch_layout(
            common,
            num_reqs=2,
            num_actual_reqs=1,
            num_input_tokens=2,
            num_actual_tokens=1,
        )


def test_query_projection_is_split_into_heads_consistently():
    query = torch.zeros((5, 12), dtype=torch.bfloat16)
    reshaped = DeepseekV41EagerAttentionImpl._reshape_query_heads(query, 4)
    assert reshaped.shape == (5, 3, 4)
    assert reshaped.data_ptr() == query.data_ptr()


def test_query_projection_rejects_partial_head():
    with pytest.raises(ValueError, match="divisible by head_dim"):
        DeepseekV41EagerAttentionImpl._reshape_query_heads(
            torch.zeros((2, 10)), 4
        )


def _ws_plan(query_lens, histories):
    qsl = [0]
    for query_len in query_lens:
        qsl.append(qsl[-1] + query_len)
    return {
        "prefill_query_start_loc": torch.tensor(qsl, dtype=torch.int32),
        "ws_local_start_pos": torch.tensor(histories, dtype=torch.int32),
    }


def _workspace_win_indices(plan, window_size=4):
    """Prefill sub-batch win indices: base helper + workspace start_pos shim."""
    qsl = plan["prefill_query_start_loc"]
    q = torch.zeros(int(qsl[-1].item()), 1, 8)
    shim = SimpleNamespace(start_pos=plan["ws_local_start_pos"])
    attn = SimpleNamespace(window_size=window_size)
    return DeepseekV41EagerAttentionImpl._get_window_topk_idxs(
        None, attn, q, shim, qsl.numel() - 1, qsl
    )


def _valid_rows(win_indices):
    return [tokens[tokens >= 0].tolist() for tokens in win_indices[:, 0]]


def test_workspace_window_rampup_without_history():
    plan = _ws_plan([5], [0])
    out = _workspace_win_indices(plan)
    assert out.shape == (5, 1, 4)
    assert out.dtype == torch.int32
    assert _valid_rows(out) == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4]]


def test_workspace_window_slides_with_history():
    plan = _ws_plan([3], [6])
    out = _workspace_win_indices(plan)
    assert _valid_rows(out) == [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]


def test_workspace_window_two_requests_keep_local_coordinates():
    plan = _ws_plan([6, 4], [3, 0])
    out = _workspace_win_indices(plan)
    assert _valid_rows(out) == [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
    ]


def test_workspace_slots_resolve_through_block_table():
    # Valid slots are local positions; ws_block_table translates them to the
    # request's workspace pages, so the resolved row must stay linear.
    block_size = 4
    plan = _ws_plan([6, 4], [3, 0])
    plan["ws_block_table"] = torch.tensor([[0, 1, 2], [3, 0, 0]], dtype=torch.int32)
    out = _workspace_win_indices(plan)
    request_of_token = [0] * 6 + [1] * 4
    page_base = {0: 0, 1: 3}
    for token in range(out.shape[0]):
        request = request_of_token[token]
        table = plan["ws_block_table"][request]
        for slot in out[token, 0].tolist():
            if slot < 0:
                continue
            resolved = int(table[slot // block_size]) * block_size + slot % block_size
            assert resolved == page_base[request] * block_size + slot
