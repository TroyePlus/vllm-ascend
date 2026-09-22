"""CPU regression executing the production non-CP output projection method."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

source = Path(__file__).resolve().parents[2] / 'vllm_ascend/attention/dsa_v1.py'
tree = ast.parse(source.read_text())
owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendDSAImpl')
method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_o_proj')
seen = []

def batchmatmul(x, weight, **kwargs):
    assert kwargs['perm_x1'] == (1, 0, 2)
    assert kwargs['perm_x2'] == (0, 1, 2)
    assert kwargs['perm_y'] == (1, 0, 2)
    seen.append(weight)
    return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)

namespace = dict(torch=torch, torch_npu=SimpleNamespace(npu_transpose_batchmatmul=batchmatmul),
                 oproj_tp_enable=lambda: False, olora_tp_enable=lambda: False)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
torch.manual_seed(1024)
for groups in (2, 4, 8):
    # Small integer-valued floats make this layout regression exact even when
    # CPU bmm chooses different kernels for contiguous and transposed weights.
    raw = torch.randint(-3, 4, (groups * 3, 16)).float()
    loaded = raw.view(groups, 3, 16).transpose(1, 2).contiguous()
    x = torch.randint(-3, 4, (7, groups, 16)).float()
    expected = torch.bmm(x.transpose(0, 1), loaded).transpose(0, 1).reshape(7, -1)
    for weight in (raw, loaded):
        obj = SimpleNamespace(n_local_groups=groups, support_fp8_attention=False,
                              wo_a=SimpleNamespace(weight=weight), wo_b=lambda y: y)
        output = torch.empty_like(expected)
        actual = namespace['_forward_o_proj'](obj, x, output)
        torch.testing.assert_close(seen[-1], loaded, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if weight is loaded:
            assert seen[-1] is loaded
print('PASS: production non-CP projection, 2D conversion and 3D identity, groups=2/4/8')

# Execute the real CP layout helper without importing the NPU runtime.
cp_source = source.parent / 'context_parallel/dsa_cp.py'
cp_tree = ast.parse(cp_source.read_text())
helper = next(n for n in ast.walk(cp_tree)
              if isinstance(n, ast.FunctionDef) and n.name == '_get_batched_wo_a_weight')
class Unquantized:
    pass
cp_namespace = dict(torch=torch, AscendUnquantizedLinearMethod=Unquantized)
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(cp_source), 'exec'), cp_namespace)
for groups in (2, 4, 8):
    expected = torch.arange(groups * 16 * 3).reshape(groups, 16, 3)
    raw_float = expected.transpose(1, 2).reshape(groups * 3, 16)
    raw_quant = expected.permute(1, 0, 2).reshape(16, groups * 3)
    for weight, method in (
        (expected, Unquantized()),
        (expected.permute(1, 0, 2), Unquantized()),
        (raw_float, Unquantized()),
        (raw_float, SimpleNamespace(quant_method=Unquantized())),
        (raw_quant, object()),
    ):
        obj = SimpleNamespace(wo_a=SimpleNamespace(weight=weight, quant_method=method))
        actual = cp_namespace['_get_batched_wo_a_weight'](obj, groups)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if weight is expected:
            assert actual is weight
print('PASS: production CP helper, loaded 3D and quantized/unquantized 2D layouts')
