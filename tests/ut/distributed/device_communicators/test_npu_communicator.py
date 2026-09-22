# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from unittest.mock import patch

import torch

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


def test_all_reduce_preserves_functional_custom_op_contract():
    comm = NPUCommunicator.__new__(NPUCommunicator)
    comm.device_group = object()
    original = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    input_ = original[1:5]
    snapshot = original.clone()

    def collective(output, group):
        assert group is comm.device_group
        output.mul_(4)

    with patch('torch.distributed.all_reduce', side_effect=collective) as mocked:
        output = comm.all_reduce(input_)

    mocked.assert_called_once()
    torch.testing.assert_close(original, snapshot)
    torch.testing.assert_close(output, snapshot[1:5] * 4)
    assert output.untyped_storage().data_ptr() != input_.untyped_storage().data_ptr()
    input_.zero_()
    torch.testing.assert_close(output, snapshot[1:5] * 4)
