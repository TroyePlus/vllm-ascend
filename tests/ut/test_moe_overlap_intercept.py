# SPDX-License-Identifier: Apache-2.0
"""Unit-test the decomposed-MoE multistream configuration boundary."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock


def _load_resolver():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py"
    tree = ast.parse(source.read_text())
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_moe_multistream_overlap"
    )
    resolver.decorator_list = []
    logger = mock.Mock()
    namespace = {"logger": logger}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_resolve_moe_multistream_overlap"], logger


def test_decomposed_moe_enables_multistream_overlap():
    resolve, logger = _load_resolver()

    actual = resolve(
        fxrt_prefill_decompose=True,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=True,
    )

    assert actual == (True, True, True)
    message = logger.info_once.call_args.args[0]
    assert "[DSV4_PREFILL_MOE_OVERLAP]" in message
    assert "enabled decomposed overlap" in message
    assert "runtime stream regions and stage events" in message


def test_eager_moe_preserves_requested_multistream_overlap():
    resolve, logger = _load_resolver()

    assert resolve(
        fxrt_prefill_decompose=False,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=True,
    ) == (True, True, True)
    assert resolve(
        fxrt_prefill_decompose=False,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=False,
    ) == (False, False, True)
    logger.warning_once.assert_not_called()


def test_decomposed_overlap_switches_are_independent():
    resolve, _ = _load_resolver()
    for shared, gate, expected in (
        (False, False, (False, False, False)),
        (True, False, (True, False, False)),
        (False, True, (False, True, True)),
        (True, True, (True, True, True)),
    ):
        assert (
            resolve(
                fxrt_prefill_decompose=True,
                requested_shared_expert=shared,
                requested_gate=gate,
                has_shared_experts=True,
            )
            == expected
        )


def test_gate_routes_before_prepare_without_consuming_prepare_state():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_gate_overlap_body")
    prepare = NS()
    ctx = NS(moe_comm_type="alltoall", moe_comm_method=NS(prepare_finalize=prepare), flash_comm_v1_enabled=True)
    seen = []

    def select(**kwargs):
        seen.append((kwargs["hidden_states"].shape[0], kwargs["gate_before_prepare"]))
        assert not vars(prepare)
        return "weights", "ids"

    namespace = dict(
        _EXTRA_CTX=ctx,
        MoECommType=NS(ALLTOALL="alltoall", MC2="mc2", FUSED_MC2="fused"),
        shared_expert_dp_enabled=lambda: True,
        select_experts=select,
        get_forward_context=lambda: NS(input_ids=None),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    layer = NS(
        _shared_experts=lambda x: "shared",
        enable_shared_expert_dp=True,
        top_k=8,
        use_grouped_topk=True,
        renormalize=False,
        topk_group=1,
        num_expert_group=1,
        custom_routing_function=None,
        scoring_func="sqrtsoftplus",
        _original_routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        moe_config=NS(num_experts=256),
        tid2eid=None,
    )
    for rows in (17, 33):
        assert namespace["_gate_overlap_body"](layer, NS(shape=(rows, 4096)), None) == ("shared", "weights", "ids")
    assert seen == [(17, True), (33, True)]


def test_hash_gate_ids_match_local_sp_rows_before_prepare():
    import torch

    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/experts_selector.py"
    tree = ast.parse(source.read_text())
    method = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_select_experts_with_fusion_ops"
    )
    # No prepare object is installed: gate routing must not depend on it.
    context = NS(input_ids=torch.tensor([10, 11, 12, 13]), flash_comm_v1_enabled=True)
    group = NS(world_size=2, rank_in_group=0)
    captured = []

    def gating(**kwargs):
        captured.append(kwargs["input_ids"].tolist())
        assert kwargs["x"].shape[0] == kwargs["input_ids"].numel()
        return torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.int32), None

    namespace = dict(
        torch=torch,
        get_forward_context=lambda: context,
        get_tp_group=lambda: group,
        MoECommType=NS(ALLGATHER="allgather"),
        split_tensor_along_first_dim=lambda x, num_partitions: torch.chunk(x, num_partitions),
        moe_gating_top_k_hash_for_prefill=gating,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    for comm in ("allgather", "alltoall"):
        context.moe_comm_type = comm
        for rank in (0, 1):
            group.rank_in_group = rank
            namespace[method.name](
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                2,
                False,
                False,
                None,
                1,
                1,
                scoring_func="sqrtsoftplus",
                tid2eid=torch.zeros(16, 2, dtype=torch.int32),
                gate_before_prepare=True,
            )
    assert captured == [[10, 11], [12, 13], [10, 11], [12, 13]]


if __name__ == "__main__":
    test_decomposed_moe_enables_multistream_overlap()
    test_eager_moe_preserves_requested_multistream_overlap()
    test_decomposed_overlap_switches_are_independent()
    test_gate_routes_before_prepare_without_consuming_prepare_state()
    test_hash_gate_ids_match_local_sp_rows_before_prepare()
    print("PASS: decomposed and eager overlap settings")
