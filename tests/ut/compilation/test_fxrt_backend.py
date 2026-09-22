#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import sys
import types
from unittest import mock

import pytest
import torch
from torch.fx import symbolic_trace
from vllm.config.compilation import CUDAGraphMode, CompilationMode

from vllm_ascend.compilation.fxrt_backend import wrap_backend_with_fxrt
from vllm_ascend.platform import _is_fxrt_backend_enabled


class Multiply(torch.nn.Module):
    def forward(self, x):
        return x * 2


def test_fxrt_backend_is_used_for_string_backend(tmp_path, monkeypatch):
    calls = []

    def fxrt_backend(gm, example_inputs):
        calls.append(gm)
        return gm.forward

    fxrt_module = types.ModuleType("fxrt")
    fxrt_torch_module = types.ModuleType("fxrt.torch")
    fxrt_torch_module.backend = fxrt_backend
    monkeypatch.setitem(sys.modules, "fxrt", fxrt_module)
    monkeypatch.setitem(sys.modules, "fxrt.torch", fxrt_torch_module)

    backend = wrap_backend_with_fxrt("inductor", tmp_path, "test/model")
    compiled_model = torch.compile(Multiply(), backend=backend, fullgraph=True)

    assert torch.equal(compiled_model(torch.ones(2)), torch.full((2,), 2.0))
    assert len(calls) == 1
    dumps = list(tmp_path.glob("fx_graph_test_model_pid*.txt"))
    assert len(dumps) == 1
    assert "==== raw graph ====" in dumps[0].read_text(encoding="utf-8")


def test_fxrt_wrapper_passes_callable_backend_without_dump():
    calls = []

    def custom_backend(gm, example_inputs):
        calls.append((gm, example_inputs))
        return gm.forward

    backend = wrap_backend_with_fxrt(custom_backend, None, "test/model")
    gm = symbolic_trace(Multiply())
    example_inputs = [torch.ones(2)]

    compiled_fn = backend(gm, example_inputs)
    assert torch.equal(compiled_fn(example_inputs[0]), torch.ones(2) * 2)
    assert calls == [(gm, example_inputs)]


@pytest.mark.parametrize(
    ("mode", "backend", "cudagraph_mode", "enabled", "expected"),
    [
        (CompilationMode.STOCK_TORCH_COMPILE, "inductor", CUDAGraphMode.NONE, True, True),
        (CompilationMode.STOCK_TORCH_COMPILE, "inductor", CUDAGraphMode.NONE, False, False),
        (CompilationMode.STOCK_TORCH_COMPILE, "inductor", CUDAGraphMode.PIECEWISE, True, False),
        (CompilationMode.DYNAMO_TRACE_ONCE, "inductor", CUDAGraphMode.NONE, True, False),
        (CompilationMode.STOCK_TORCH_COMPILE, "eager", CUDAGraphMode.NONE, True, False),
    ],
)
def test_direct_fxrt_backend_config_predicate(
    mode, backend, cudagraph_mode, enabled, expected, monkeypatch
):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FXRT_BACKEND", "1" if enabled else "0")
    vllm_config = mock.MagicMock()
    vllm_config.compilation_config.mode = mode
    vllm_config.compilation_config.backend = backend
    vllm_config.compilation_config.cudagraph_mode = cudagraph_mode

    assert _is_fxrt_backend_enabled(vllm_config) is expected
