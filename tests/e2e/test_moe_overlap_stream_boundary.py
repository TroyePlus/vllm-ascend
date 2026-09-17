# SPDX-License-Identifier: Apache-2.0
"""One NPU: verify FXRT stream-region ordering without a model checkpoint.

Run with the same FXRT/torch_npu environment as the model service. The small
fixture substitutes only the layer arithmetic; real streams/events and the
production custom-op implementations execute through FXRT.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch_npu  # noqa: F401
from fxrt.torch import backend


def main():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/overlap_region.py"
    spec = importlib.util.spec_from_file_location("overlap_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    gate_stream = torch.npu.Stream()
    shared_stream = torch.npu.Stream()
    seen_streams = []

    class Layer:
        def __init__(self):
            self.gate_stream = gate_stream

        def _gate_overlap_body(self, x, logits):
            seen_streams.append(torch.npu.current_stream().npu_stream)
            scores, ids = torch.topk(logits, 2)
            return x * 2, scores, ids.to(torch.int32)

        def _forward_shared_experts(self, x, events):
            assert events.before_dispatch == 11 and events.before_gmm2 == 12
            shared_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(shared_stream):
                seen_streams.append(torch.npu.current_stream().npu_stream)
                result = x * 3
            torch.npu.current_stream().wait_stream(shared_stream)
            return result

    module._layer = lambda name: Layer()

    def run(x, logits):
        hidden = x * 2
        shared, scores, ids = module.gate_overlap(hidden, logits, [x], "layer", 2)
        routed = hidden + 8
        module.overlap_wait([hidden, logits], 0)
        other = module.shared_overlap(x, routed, [x], "layer", [9, 10, 11, 12, 13], 10.0)
        return shared + other + routed, scores, ids

    graphs = []

    def capture(gm, inputs):
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        assert any("fxrt_moe_gate_overlap" in t for t in targets)
        assert any("fxrt_moe_shared_overlap" in t for t in targets)
        assert any("fxrt_moe_overlap_wait" in t for t in targets)
        assert not any("set_stream" in t or "get_external_object" in t for t in targets)
        gate_node = next(n for n in gm.graph.nodes if "fxrt_moe_gate_overlap" in str(n.target))
        wait_node = next(n for n in gm.graph.nodes if "fxrt_moe_overlap_wait" in str(n.target))
        assert tuple(wait_node.args[0]) == tuple(gate_node.args[:2])
        graphs.append(gm)
        return backend(gm, inputs)

    compiled = torch.compile(run, backend=capture, fullgraph=True, dynamic=True)
    stub = SimpleNamespace(FusedMoEEvents=SimpleNamespace)
    with mock.patch.dict(
        "sys.modules",
        {
            "vllm_ascend.ops.fused_moe.fused_moe": stub,
            "vllm_ascend.ops.fxrt_side_effects": SimpleNamespace(_stream_from_index=lambda index: gate_stream),
        },
    ):
        for rows in (17, 33, 65, 17):
            x = torch.arange(rows * 16, device="npu", dtype=torch.float32).reshape(rows, 16)
            logits = x + 0.5
            output, scores, ids = compiled(x, logits)
            torch.npu.synchronize()
            torch.testing.assert_close(output, x * 9 + 8, rtol=0, atol=0)
            expected_scores, expected_ids = torch.topk(logits, 2)
            torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
            torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    assert len(graphs) == 1, len(graphs)
    assert set(seen_streams) == {gate_stream.npu_stream, shared_stream.npu_stream}
    print("PASS: FXRT fullgraph, one dynamic graph, gate/shared execute on their own streams")


if __name__ == "__main__":
    main()
