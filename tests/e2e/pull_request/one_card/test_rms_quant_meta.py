# SPDX-License-Identifier: Apache-2.0
"""Run directly to check the Python Meta correction against the NPU kernel."""

import torch
import torch_npu  # noqa: F401


def test_rms_quant_meta():
    # Match worker registration order without initializing unrelated ops.
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    import vllm_ascend.ops.rms_quant_meta  # noqa: F401

    op = torch.ops._C_ascend.npu_rms_norm_dynamic_quant
    graphs = []

    def model(x, gamma):
        q, scale = op(x, gamma, epsilon=1e-6)
        # Regression: wrong Meta previously selected a second quantization.
        if q.dtype == torch.bfloat16:
            q, scale = torch_npu.npu_dynamic_quant(q)
        return q, scale

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    for dtype in (torch.bfloat16, torch.float16):
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        gamma = torch.ones(1024, device="npu", dtype=dtype)
        for tokens in (8, 16, 32):
            x = torch.randn(tokens, 1024, device="npu", dtype=dtype)
            expected = model(x, gamma)
            meta = op(x.to("meta"), gamma.to("meta"), epsilon=1e-6)
            actual = compiled(x, gamma)
            for result, inferred, reference in zip(actual, meta, expected):
                assert inferred.shape == reference.shape
                assert inferred.dtype == reference.dtype
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert expected[0].dtype == torch.int8
        assert expected[1].dtype == torch.float32
    assert graphs
    assert all("npu_dynamic_quant" not in gm.code for gm in graphs)
    torch.npu.synchronize()


if __name__ == "__main__":
    torch.npu.set_device(0)
    test_rms_quant_meta()
    print("PASS: RMS Meta, symbolic shapes, and compiled/eager outputs agree")
