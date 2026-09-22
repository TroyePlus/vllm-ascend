# SPDX-License-Identifier: Apache-2.0
"""Exercise the migrated shared-expert constructor without importing NPU kernels."""

import ast
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_constructor():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/shared_experts.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSharedExperts")
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), constructor],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    logger = mock.Mock()
    config = SimpleNamespace(multistream_overlap_shared_expert=False, enable_shared_expert_dp=True)
    gate = mock.Mock(return_value=False)
    namespace = {
        "logger": logger,
        "get_ascend_config": lambda: config,
        "fxrt_moe_prefill_decompose_enabled": gate,
        "SituAndMul": type("SituAndMul", (), {}),
        "wraps": wraps,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["__init__"], config, gate, logger


def test_shared_expert_overlap_configuration_boundary():
    for decompose in (False, True):
        for requested in (False, True):
            initialize, config, gate, logger = _load_constructor()
            config.multistream_overlap_shared_expert = requested
            gate.return_value = decompose
            executor = SimpleNamespace(validate_consistency=mock.Mock())
            layer = SimpleNamespace(gate_up_proj=SimpleNamespace(input_size=16), act_fn=object())
            moe_config = SimpleNamespace(
                hidden_dim=16, in_dtype="bf16", swiglu_limit=None,
                swiglu_alpha=None, swiglu_beta=None, is_sequence_parallel=True,
            )
            process = mock.Mock(return_value="loaded")
            quant_method = SimpleNamespace(process_weights_after_loading=process)
            initialize(executor, layer, moe_config, "w8a8", quant_method)
            effective = requested and not decompose
            assert executor.multistream_overlap is effective
            assert executor.weights_replicated is True
            assert quant_method.process_weights_after_loading("weights") == "loaded"
            process.assert_called_once_with("weights")
            assert executor.validate_consistency.call_count == int(effective)
            if requested and decompose:
                assert "[DSV4_PREFILL_MOE_OVERLAP]" in logger.warning_once.call_args.args[0]
            else:
                logger.warning_once.assert_not_called()


if __name__ == "__main__":
    test_shared_expert_overlap_configuration_boundary()
    print("PASS: migrated shared-expert overlap boundary (four configurations)")
