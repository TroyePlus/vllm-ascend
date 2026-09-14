"""Standalone: python tests/ut/test_moe_route_audit.py (no NPU allocation)."""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm_ascend import moe_route_audit as audit


class RouteAuditTest(unittest.TestCase):
    def setUp(self):
        group = NS(rank_in_group=0, world_size=4)
        config = NS(enable_prefill_mc2=False, enable_fused_mc2=0, eplb_config=NS(config={}))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("get_dp_group", "get_tp_group", "get_ep_group"):
            self.stack.enter_context(patch("vllm.distributed." + name, return_value=group))
        self.stack.enter_context(patch("vllm_ascend.ascend_config.get_ascend_config", return_value=config))
        self.stack.enter_context(patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=32))
        self.stack.enter_context(patch("vllm_ascend.utils.get_ascend_device_type", return_value="A2"))
        self.stack.enter_context(patch.object(audit.envs, "VLLM_ASCEND_MOE_AUDIT", True))
        self.stack.enter_context(patch.object(audit.envs, "VLLM_ASCEND_MOE_AUDIT_PROFILE", False))
        self.stack.enter_context(patch.object(audit.envs, "VLLM_ASCEND_MOE_AUDIT_LIMIT", 2))
        self.config = NS(_moe_audit_state=dict(seen=audit.Counter(), sequence=0, config_printed=True))
        self.ctx = NS(
            num_tokens=256,
            max_tokens_across_dp=256,
            pad_size=0,
            in_profile_run=False,
            is_draft_model=False,
            moe_comm_type=NS(name="ALLGATHER"),
        )

    def run_forward(self):
        return audit.audit_forward(self.config, self.ctx, 256, 256, None, False)

    def test_bounded_and_disabled(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for _ in range(5):
                with self.run_forward():
                    pass
        events = [json.loads(line.split("[MOE_AUDIT] ")[1])["event"] for line in output.getvalue().splitlines()]
        self.assertEqual(events, ["SELECT", "END", "SELECT", "END"])
        with patch.object(audit.envs, "VLLM_ASCEND_MOE_AUDIT", False), self.run_forward():
            pass
        self.assertEqual(self.config._moe_audit_state["sequence"], 5)

    def test_exception_preserved_and_no_tensor_reads(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "original"),
            self.run_forward(),
        ):
            raise ValueError("original")

        class TensorLike:
            def item(self):
                raise AssertionError("must not read tensor")

        self.assertEqual(audit._plain(TensorLike()), "TensorLike")

    def test_fullgraph_and_profiler_boundary(self):
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward

        fn = torch.compile(lambda x: x * 2 + 1, backend=backend, fullgraph=True)
        with patch.object(audit.envs, "VLLM_ASCEND_MOE_AUDIT_PROFILE", True), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                with self.run_forward():
                    torch.testing.assert_close(fn(torch.ones(4)), torch.full((4,), 3.0))
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
