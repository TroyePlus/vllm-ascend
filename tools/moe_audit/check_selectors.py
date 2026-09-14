"""Run the actual selector AST with simulated topology, never NPU execution."""

import ast
import enum
from pathlib import Path
from types import SimpleNamespace as NS

source = Path(__file__).resolve().parents[2] / "vllm_ascend/ascend_forward_context.py"
tree = ast.parse(source.read_text())
names = {"_select_a2_moe_comm_method", "_select_a3_moe_comm_method", "select_moe_comm_method"}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
for fn in functions:
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
soc_type = enum.Enum("AscendDeviceType", "A2 A3 A5 _310P")
comm_type = enum.Enum("MoECommType", "ALLGATHER MC2 ALLTOALL FUSED_MC2")
namespace = dict(
    AscendDeviceType=soc_type,
    MoECommType=comm_type,
    get_ep_group=lambda: NS(world_size=8),
    is_moe_model=lambda _: True,
    get_mc2_tokens_capacity=lambda: 32,
    get_ascend_config=lambda: NS(enable_fused_mc2=0),
    logger=NS(debug=lambda *a, **k: None),
)
exec(compile(module, str(source), "exec"), namespace)
config = NS(
    parallel_config=NS(enable_expert_parallel=True, world_size_across_dp=8, pipeline_parallel_size=1),
    model_config=NS(get_num_experts=lambda: 256, hf_text_config=NS(n_routed_experts=256, moe_quantize="w8a8_dynamic")),
)
for soc in (soc_type.A2, soc_type.A3):
    namespace["get_ascend_device_type"] = lambda soc=soc: soc
    for tokens in (32, 33, 256, 2048, 8192):
        actual = namespace["select_moe_comm_method"](tokens, config).name
        expected = "ALLGATHER" if soc == soc_type.A2 else ("MC2" if tokens <= 32 else "ALLTOALL")
        assert actual == expected, (soc, tokens, actual, expected)
        print("SELECTOR_UNIT_ONLY", soc.name, "tokens=" + str(tokens), actual)
