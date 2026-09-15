# 7.4.1 FXRT TTFT comparison

Measured on 2026-09-15 in 131's `dsv4-pr72-final`, A2 devices 0--3,
DP=2, TP=2, EP=true, DSA-CP=true, enable_prefill_mc2=true.

| Mode | Successful requests | Input tokens | Mean TTFT (ms) | Population stddev (ms) |
| --- | --- | --- | --- | --- |
| Direct FXRT, opaque DSA/MoE | 100/100 | 8 | 120.898792 | 9.516440 |
| Direct FXRT, decomposed DSA/MoE | 100/100 | 8 | 105.622590 | 8.289631 |

Each mode used 10 warmup requests followed by 100 measured requests,
concurrency 1, output length 1, seed 1024. The statistics use the 100 individual
`ttfts` in `bench/samples.json`, in seconds converted to milliseconds;
`statistics.pstdev` uses denominator N. Actual server-reported `input_lens`
were all 8. `/v1/completions` avoids adding a chat template to the input.

Both modes use `fullgraph=True`, `dynamic=True`, compilation mode 1,
cudagraph NONE, direct external FXRT. Inductor FXRT and AscendC fusion are
disabled. The full-mode FX graphs contain `vllm.dsa_forward` and
`vllm.moe_forward_shared`; the split-mode graphs expand these calls and contain
`vllm_ascend.fxrt_moe_gating_top_k_hash`. Split here means exposing operator
internals, not enabling Dynamo graph breaks. Each active DP0 worker compiled
two graphs during warmup; no further backend entries were recorded during
the measured requests. Requests target the DP0 HTTP endpoint; this is not
a simultaneous load test of both DP endpoints or an end-to-end P/D proxy test.

## Code and model provenance

- PR #16371 HEAD and container model source: `f75236a7c95161d18f9039719661f58109c0374d`.
- vLLM source: `f5ffef0859241759035b4cc3db0360b9c1d41048`.
- Torch `2.10.0+cpu`, torch_npu `2.10.0.post2`, FXRT `0.1.dev0+5a29416`.
- Model is the existing reduced dummy model, not production weights.
- `VLLM_ASCEND_FXRT_DUMMY_QUANT=1` is enabled identically in both modes.
- The model source and installed FXRT were not modified for this comparison.
- `VLLM_VERSION=0.23.0` selects the compatibility path matching the checked-out
  vLLM source; its installed version label is `0.25.1+empty`.
- Memory utilization is 0.45, model length 4096, batch-token limit 8192,
  max sequences 16, MTP=1, Mooncake producer with prefill DP2/TP2 and decode
  DP8/TP1 metadata. No A3 route forcing is enabled.

These are reduced-model measurements on a shared A2 host, not production
accuracy or A3 performance results. Do not interpret a single pair as a
statistically established speedup.

## Reproduce inside the container

The scripts are in `/workspace/dsv4/ttft741`; tracked copies are in
`tools/moe_audit`. Check available cards before starting. Run one mode at a time.

```bash
export MOE_AUDIT_DUMP=/workspace/dsv4/logs/ttft741-full/fx_dump
bash /workspace/dsv4/ttft741/start_ttft_mode.sh full '0,1;2,3' \
  /workspace/dsv4/logs/ttft741-full
```

In another shell, after `/health` returns 200:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash /workspace/dsv4/ttft741/run_ttft_compare.sh 127.0.0.1 18990 \
  /workspace/dsv4/logs/ttft741-full/bench
```

Stop the full-mode service and release its workers, then repeat with `split`
and a different log/dump directory. The two services must not run together.
The launcher accepts `P_MODEL`, `P_LOAD_FORMAT`, `P_MEMORY_UTIL`, `P_NIC`,
`P_LOCAL_IP`; defaults describe this container's dummy test setup. It launches
HTTP ports 18990/18991, DP RPC 18772 and KV port 30773. Device groups must be
quoted because the separator is a semicolon.

## Evidence

- Container: `/workspace/dsv4/logs/ttft741-{full,split}/`.
- 127 host: `/home/liyizhan/dsv4/ttft741-{full,split}/`.
- Each contains service rank logs, FX graphs, benchmark log, detailed JSON,
  and `bench/result.txt`.
- Both test services were stopped by restarting the dedicated container
  after collection; its process list returned to only `sleep infinity`.

Validation: shell syntax checks and `git diff --check` passed. Repository-wide
`bash format.sh ci` could not run because `pre-commit` is not installed.
