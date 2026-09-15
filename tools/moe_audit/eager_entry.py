"""Test-only CLI entry: exercise decomposition without torch.compile.

Run with python -m tools.moe_audit.eager_entry serve ... --enforce-eager.
Top-level installation is intentional: multiprocessing spawn re-executes this
module in workers before model_runner imports the configuration helper.
"""

import os

from vllm_ascend import utils


def configure_eager_decompose(config):
    assert config.model_config.enforce_eager, "This entry is only for eager tests"
    assert int(config.compilation_config.mode) == 0, "Compilation must be disabled"
    kv = config.kv_transfer_config
    producer = kv is None or (kv.is_kv_producer and not kv.is_kv_consumer)
    active = producer and os.getenv("VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL") == "1"
    utils._FXRT_PREFILL_DECOMPOSE_ACTIVE = active
    print(f"[EAGER_AUDIT] pid={os.getpid()} decomposition={int(active)} compile_mode=0", flush=True)
    return active


utils.configure_fxrt_prefill_decompose = configure_eager_decompose

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
