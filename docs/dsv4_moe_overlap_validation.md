# DeepSeek V4 prefill MoE overlap

## Scope

This branch builds on `030ae1636b0359f28782b338b4721e0c33ea4a8c`.
For vLLM 0.23.0, both the earlier `19dcffcf6` decomposition and `77155f5c1`
disabled shared-expert/gate overlap in `fused_moe_0_23_0.py`. The other version
implementation in `fused_moe.py` is not evidence of 0.23.0 behavior.

The base branch keeps that restriction and logs requested versus effective
settings. This development branch restores runtime stream scopes and stage
events for the decomposed path. It does not change the DSA/MoE switch contract.

## Configuration

Use `VLLM_VERSION=0.23.0`, the direct FXRT backend, and
`VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_MOE=1`. The existing additional-config
options `multistream_overlap_shared_expert` and `multistream_overlap_gate`
select overlap. Their default remains false. DSA decomposition is independent.

Graph boundaries:

| Operator | Runtime behavior |
| --- | --- |
| `vllm_ascend::fxrt_alltoall_routed_experts` | Original dispatch, routed MLP and combine; ragged receive buffers remain internal. |
| `vllm_ascend::fxrt_moe_shared_overlap` | Shared experts on their native stream, with original stage-event waits. |
| `vllm_ascend::fxrt_moe_gate_overlap` | Shared/gate arithmetic on the native gate stream. |
| `vllm_ascend::fxrt_moe_overlap_gather` | Original EP gather/unpadding, including the quant communication stream. |
| `vllm_ascend::fxrt_moe_overlap_reduce` | Original EP padding/reduce-scatter, returning the pre-gather local token layout. |
| `vllm_ascend::fxrt_moe_overlap_wait` | Wait for a side stream and retain its graph-owned input tensors until that wait. |

Stream/Event objects stay outside the FX ABI. Preallocated integer handles
identify before-routing, after-routing, dispatch, GMM2 and combine events.
Shared/routed tensors and shared weights are explicit graph inputs. These
operators do not mutate those tensor inputs.

A scalar-only stream wait is insufficient for graph-owned temporary buffers:
the memory recycler cannot infer which tensors a side stream still reads.
The tensor-input wait retains gate and quant-gather inputs until the existing
join points; it does not move the waits earlier or add a device-wide barrier.

Gate hash routing executes before prepare. It uses the local token-ID shard
matching its local router rows, then gathers routing outputs when required.
The ordinary post-prepare routing path retains its existing ID handling.

AllGather unpadding uses scheduler CPU metadata. The total gathered token
count is computed before tracing, without a device-to-host transfer. Reduce
uses the original local SP row count, not global rows divided only by TP:
when DP > 1, those shapes differ. Both runtime boundaries check actual rows
against the shape contract.

The existing AllToAll region remains limited to static W8A8 without LoRA,
dynamic EPLB, fused scale-bias or offsets. Its original asynchronous
collectives and waits are retained; this branch does not replace them with
synchronous collectives or remove CPU split-list synchronization.

## Validation

Validated on four Ascend 910B cards with DP2 x TP2, EP4, DSA-CP, seeded
nonzero four-layer dummy weights and twelve fixed natural-language requests
(three prompt lengths, four repeats). The service is stopped after each run.

| Comparison | Result |
| --- | --- |
| Base FXRT, overlap disabled vs this branch FXRT, overlap disabled | API and complete first-token hidden/logit tensors exactly equal. |
| Shared-only overlap eager vs FXRT | API and complete first-token hidden/logit tensors exactly equal. |
| Gate plus shared overlap eager vs FXRT | API and complete first-token hidden/logit tensors exactly equal. |

`tests/ut/test_moe_overlap_intercept.py` verifies the independent switch
contract and `tests/ut/test_alltoall_region_payload.py` verifies the stable
event payload. The standalone
`tests/e2e/test_moe_overlap_stream_boundary.py` exercises real NPU streams and
FXRT fullgraph execution with a small arithmetic fixture.

This is functional and numerical validation of the tested DP2 x TP2 dummy
configuration, not a full-weight A3 eight-card numerical or performance
guarantee. Test artifacts are kept under `/workspace/audit719/results` in the
validation container; source changes are kept in this repository, not
installed-package edits.
