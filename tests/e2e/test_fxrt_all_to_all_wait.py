# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=2 --master-addr=127.0.0.1.

CPU/Gloo regression for the opaque async collective + wait boundary.
No NPU allocation is required; torch_npu must be importable.
"""
import torch
import torch.distributed as dist

from vllm_ascend.ops.fused_moe.comm_utils import _fxrt_all_to_all_wait


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    group_name = dist.group.WORLD.group_name

    def forward(x, recv, send, capacity):
        return _fxrt_all_to_all_wait(x, recv, send, capacity, group_name) + 1

    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=True)
    try:
        for n in (2, 6, 18):
            for unequal in (False, True):
                send = [0, n] if unequal and rank == 0 else [n // 2, n // 2]
                recv = ([0, n // 2] if rank == 0 else [n, n // 2]) if unequal else send
                x = torch.arange(n, dtype=torch.float32) + rank * 100
                args = (x, torch.tensor(recv), torch.tensor(send), sum(recv))
                reference = forward(*args)
                actual = compiled(*args)
                torch.testing.assert_close(actual, reference)
        print(f"PASS rank={rank}: equal/unequal splits, three lengths", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
