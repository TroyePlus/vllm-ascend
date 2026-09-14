"""Opt-in, bounded MRv1 diagnostics outside the compiled model forward.

No tensor values are read, no routes are overridden, and no graph nodes are
inserted. CPU profiling observes host operator submissions, not NPU completion.
"""

import json
import os
from collections import Counter
from contextlib import contextmanager

import torch

from vllm_ascend import envs


def _emit(event, **fields):
    print(
        "[MOE_AUDIT] " + json.dumps(dict(event=event, pid=os.getpid(), **fields), sort_keys=True, default=str),
        flush=True,
    )


def _plain(value):
    # Never call item(), to(cpu), or bool(Tensor) for diagnostics.
    return value if value is None or type(value) in (int, float, bool, str) else type(value).__name__


def _metadata(metadata):
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    return {
        key: _plain(getattr(metadata, key, None))
        for key in ("num_prefills", "num_prefill_tokens", "num_decode_tokens", "num_actual_tokens")
    }


@contextmanager
def audit_forward(config, ctx, selected_tokens, actual_tokens, metadata, skip_compiled):
    if not envs.VLLM_ASCEND_MOE_AUDIT:
        yield
        return
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ascend_forward_context import get_mc2_tokens_capacity
    from vllm_ascend.utils import get_ascend_device_type

    dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
    state = getattr(config, "_moe_audit_state", None)
    if state is None:
        state = dict(seen=Counter(), sequence=0, config_printed=False)
        config._moe_audit_state = state
    seen = state["seen"]
    ranks = dict(dp=dp.rank_in_group, tp=tp.rank_in_group, ep=ep.rank_in_group)
    ac = get_ascend_config()
    if not state["config_printed"]:
        hf = config.model_config.hf_text_config
        pc, cc, sc = config.parallel_config, config.compilation_config, config.scheduler_config
        _emit(
            "CONFIG",
            **ranks,
            dp_size=dp.world_size,
            tp_size=tp.world_size,
            ep_size=ep.world_size,
            soc=str(get_ascend_device_type()),
            ep_enabled=pc.enable_expert_parallel,
            experts=getattr(hf, "n_routed_experts", getattr(hf, "num_experts", None)),
            topk=getattr(hf, "num_experts_per_tok", None),
            quant=getattr(hf, "moe_quantize", getattr(hf, "quantize", None)),
            model_quant=type(config.quant_config).__name__,
            mtp=config.speculative_config is not None,
            dynamic_eplb=ac.eplb_config.config.get("dynamic_eplb"),
            prefill_mc2=ac.enable_prefill_mc2,
            fused_mc2=ac.enable_fused_mc2,
            capacity=get_mc2_tokens_capacity(),
            max_batch=sc.max_num_batched_tokens,
            max_seqs=sc.max_num_seqs,
            capture_max=cc.max_cudagraph_capture_size,
            eager=config.model_config.enforce_eager,
            compile_mode=str(cc.mode),
            backend=str(cc.backend),
            external=os.getenv("VLLM_EXTERNAL_FX_BACKEND"),
            decompose=os.getenv("VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL", "0"),
            additional=config.additional_config,
            flashcomm1=os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1"),
            force_alltoall=os.getenv("DSV4_TEST_FORCE_ALLTOALL"),
            force_mc2=os.getenv("DSV4_TEST_FORCE_MC2"),
        )
        state["config_printed"] = True
    state["sequence"] += 1
    sequence = state["sequence"]
    fields = dict(
        **ranks,
        seq=sequence,
        local=ctx.num_tokens,
        selector_tokens=selected_tokens,
        actual=_plain(actual_tokens),
        max_dp=ctx.max_tokens_across_dp,
        pad=ctx.pad_size,
        padded=getattr(ctx, "padded_num_tokens", None),
        profile=ctx.in_profile_run,
        metadata=metadata is not None,
        draft=ctx.is_draft_model,
        skip_compiled=skip_compiled,
        route=getattr(ctx.moe_comm_type, "name", None),
        capacity=get_mc2_tokens_capacity(),
        **_metadata(metadata),
    )
    key = tuple(str(fields[k]) for k in fields if k not in ("seq",))
    limit = envs.VLLM_ASCEND_MOE_AUDIT_LIMIT
    sample = (key in seen or len(seen) < limit) and seen[key] < 2
    if not sample:
        yield
        return
    seen[key] += 1
    fields["sample"] = seen[key]
    _emit("SELECT", **fields)
    profiler = None
    if envs.VLLM_ASCEND_MOE_AUDIT_PROFILE:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        profiler.__enter__()
    outcome = "returned"
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        _emit("ERROR", **ranks, seq=sequence, error=outcome, message=str(exc)[:700])
        raise
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            counts = Counter()
            for event in profiler.key_averages():
                name = event.key
                if any(
                    word in name.lower()
                    for word in (
                        "moe_distribute",
                        "moe_dispatch",
                        "dispatch_ffn",
                        "dispatch_gmm",
                        "alltoall",
                        "all_to_all",
                        "allgather",
                        "all_gather",
                        "moe_init_routing",
                        "vllm::moe",
                        "vllm::dsa",
                        "vllm::mla",
                    )
                ):
                    counts[name] += event.count
            _emit("OPS", **ranks, seq=sequence, scope="host_may_include_tracing", sample=seen[key], counts=dict(counts))
        _emit("END", **ranks, seq=sequence, result=outcome)
