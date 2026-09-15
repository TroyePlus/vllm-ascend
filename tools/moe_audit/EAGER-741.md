# 7.4.1 Eager comparison

Eager does not capture a graph. Here `full` means retaining opaque DSA/MoE
custom-op boundaries, and `split` means executing their exposed implementations
in Python. Neither uses torch.compile, FXRT, Inductor, AscendC fusion or ACL graphs.

## Measurements (2026-09-15)

| PR mode | Measured successes | Input tokens | TTFT mean (ms) | Population standard deviation (ms) |
| --- | --- | --- | --- | --- |
| Eager, opaque | 100/100 | 8 | 124.737039 | 29.110920 |
| Eager, decomposed | 100/100 | 8 | 111.993186 | 11.380973 |

Both runs use 10 warmup requests, then 100 measured requests, concurrency 1,
output length 1, seed 1024, and `/v1/completions`. All actual reported input
lengths are 8. Each server log contains 110 successful completion requests.
Statistics are calculated from individual `ttfts` in the detailed JSON,
not from the printed mean and median. These are single runs on a shared host,
not proof of a statistically reliable speedup.

The 131 container is `dsv4-pr72-final`; devices 2,3 and 6,7 form DP2/TP2,
EP=4. DSA-CP, shared-expert DP and prefill MC2 are enabled; native A2 route
selection is retained. The reduced dummy model and existing
`VLLM_ASCEND_FXRT_DUMMY_QUANT=1` compatibility switch match the previous
FXRT comparison. Model length 4096, max batch tokens 8192, max sequences 16,
memory utilization 0.45, MTP=1, Mooncake producer configuration are unchanged.
Requests go to DP0's HTTP endpoint; this is not a P/D proxy latency measurement.

Model implementation equals PR #16371 HEAD `f75236a7c`; vLLM is `f5ffef085`.
Torch is `2.10.0+cpu`, torch_npu `2.10.0.post2`. FXRT remains installed but
is not executed. No site-packages edits or model implementation edits were made.

## Why a test entry is needed

`configure_fxrt_prefill_decompose` normally disables decomposition when
compilation mode is 0. Setting the decomposition environment variable together
with `--enforce-eager` alone would compare the same opaque path twice.

`eager_entry.py` is a test-only CLI bootstrap. It overrides only this
configuration resolver before workers load their models, asserts eager mode
and compilation mode 0, and retains the producer-role check. Worker startup
prints `[EAGER_AUDIT] ... decomposition=0/1 compile_mode=0`. Both measurements
verified the expected flag on all four workers and no external FX backend entry.
This override is confined to this CLI process; ordinary vllm serve is unchanged.

## Reproduce and stop

From `/vllm-workspace/vllm-ascend` in the container, after checking free devices:

```bash
P_EAGER=1 bash tools/moe_audit/start_ttft_mode.sh full '2,3;6,7' \
  /workspace/dsv4/logs/eager741-full
```

In another shell, wait for health then run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash tools/moe_audit/run_ttft_compare.sh 127.0.0.1 18990 \
  /workspace/dsv4/logs/eager741-full/bench
python tools/moe_audit/stop_ttft.py \
  /workspace/dsv4/logs/eager741-full/launcher.pids
```

Repeat with `split` and a separate log directory only after releasing the
previous service. `P_EAGER_RAW=1` selects ordinary vllm serve without the test
bootstrap. `P_ASCEND_SOURCE` selects a different source worktree for historical
comparison. Do not use RAW for decomposed eager: the normal resolver disables it.

## Why disabling decomposition is not the same as resetting

The decomposition commit is `19dcffcf6`; its parent is `d20ac15e7`.
Besides conditional decomposition, it unconditionally replaces
`_C_ascend.npu_moe_init_routing_custom` with
`torch_npu.npu_moe_init_routing_v2` in `device/device_op.py`. Later commits also
change compatibility behavior. Therefore disabling decomposition cannot be
claimed equivalent to resetting to the parent.

A detached worktree `/vllm-workspace/vllm-ascend-before-decompose` at `d20ac15e7`
was created for the historical control, without resetting the active branch.
Its two extension libraries are symlinks to the current container's libraries;
this compares old Python code with the available binary environment, not a
reconstruction of an old wheel.

The historical control failed during startup profiling, before accepting any
requests: `aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so, or
libopapi.sonot found.` The stack points into this worktree's original `hc_pre`
implementation. No valid historical TTFT is available. Replacing that kernel
or copying newer dummy compatibility into it would cease to be an unmodified
parent-commit control. This limitation remains open; only the two PR-mode
measurements above are complete.

## Evidence locations

- Container: `/workspace/dsv4/logs/eager741-{full,split,before}`.
- 127: `/home/liyizhan/dsv4/eager-full` and `eager-split`.
- Each successful run includes rank logs, `bench/samples.log`, detailed
  `bench/samples.json`, and `bench/result.txt`.
- Historical failure logs: `/home/liyizhan/dsv4/eager-before` on 127.
- Test worker processes were released after each run. PID1 retains defunct
  children, which do not hold NPU allocations; the user's shell/editor was
  preserved, so the container was not restarted.
