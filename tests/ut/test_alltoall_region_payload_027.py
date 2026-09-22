"""Validate the migrated region against real 0.27 payload classes on CPU.

Only expert execution/communication is substituted: no NPU collectives here.
"""
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe.alltoall_region import run_alltoall_routed_region
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType

def test_real_payload_and_fake_contract():
    method = SimpleNamespace(moe_config=SimpleNamespace(num_local_experts=3, swiglu_limit=10.0))
    seen = []

    def execute(instance, payload):
        assert instance is method
        assert payload.quant.quant_type == QuantType.W8A8
        assert payload.activation == MoEActivation.SILU
        assert instance.moe_config.swiglu_limit == 10.0
        seen.append(payload)
        return comm.FusedExpertsResult(routed_out=payload.hidden_states + 1,
            expert_tokens=torch.tensor([2, 1, 0], dtype=torch.int64))

    weight = torch.ones(3, 8, 8, dtype=torch.int8)
    scale = torch.ones(3, 8)
    for count in (1, 7, 19):
        x = torch.arange(count * 8, dtype=torch.float32).reshape(count, 8)
        payload = build_fused_experts_input(hidden_states=x,
            topk_weights=torch.ones(count, 1), topk_ids=torch.zeros(count, 1, dtype=torch.int32),
            w1=weight, w2=weight, w1_scale=scale, w2_scale=scale,
            quant_type=QuantType.W8A8, dynamic_eplb=False, comm_quant_mode=2)
        with patch.object(comm, 'get_moe_comm_method', return_value=method), \
             patch.object(comm.MoECommMethod, 'fused_experts', execute):
            result = run_alltoall_routed_region(method, payload)
            torch.testing.assert_close(result.routed_out, x + 1)
            assert result.expert_tokens.tolist() == [2, 1, 0]
            assert seen[-1].weights.w1[0].data_ptr() == weight.data_ptr()
            assert seen[-1].routing.pertoken_scale is None
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake = run_alltoall_routed_region(method, payload)
            assert fake.routed_out.shape == x.shape and fake.routed_out.dtype == x.dtype
            assert fake.expert_tokens.shape == (3,) and fake.expert_tokens.dtype == torch.int64
    print('PASS: real 0.27 payload reconstruction, tensor inputs, output and FakeTensor contracts')


if __name__ == '__main__':
    test_real_payload_and_fake_contract()
