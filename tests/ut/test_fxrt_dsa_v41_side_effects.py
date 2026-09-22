import ast
from pathlib import Path


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_v41_stage_custom_ops_are_integer_handle_boundaries():
    source = Path(__file__).parents[2] / "vllm_ascend" / "ops" / "fxrt_side_effects.py"
    names = _function_names(source)
    assert {
        "fxrt_dsa_v41_stage_begin",
        "fxrt_dsa_v41_stage_ready",
        "fxrt_dsa_v41_stage_join",
    } <= names
    text = source.read_text()
    assert '"vllm_ascend::fxrt_dsa_v41_stage_begin"' in text
    assert '"vllm_ascend::fxrt_dsa_v41_stage_ready"' in text
    assert '"vllm_ascend::fxrt_dsa_v41_stage_join"' in text


def test_v41_graph_safe_preprocess_uses_stage_boundaries():
    source = Path(__file__).parents[2] / "vllm_ascend" / "attention" / "dsa_v41.py"
    text = source.read_text()
    assert "graph_safe: bool = False" in text
    assert "graph_safe=True" in text
    assert "get_npu_stream_index(aux_stream)" in text
