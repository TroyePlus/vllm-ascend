#!/usr/bin/env bash
# Source CANN/venv before invocation. Seven positional args match production P.
set -euo pipefail
[[ $# == 7 ]] || { echo 'Usage: P.sh DEVICES HTTP_PORT DP_SIZE DP_RANK DP_ADDRESS RPC_PORT TP_SIZE' >&2; exit 2; }
: "${P_NIC:?set P_NIC to the host network interface}"
: "${P_LOCAL_IP:?set P_LOCAL_IP to the HCCL interface IP}"
: "${P_MODEL:?set P_MODEL to the model directory}"
export HCCL_IF_IP="$P_LOCAL_IP"
export GLOO_SOCKET_IFNAME="$P_NIC" TP_SOCKET_IFNAME="$P_NIC" HCCL_SOCKET_IFNAME="$P_NIC"
export VLLM_RPC_TIMEOUT=3600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204 HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=1 HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES="$1"
export VLLM_EXTERNAL_FX_BACKEND=fxrt VLLM_USE_AOT_COMPILE=0
export VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL=${VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL:-1}
# Real weights: do not implicitly activate dummy-only numeric compatibility.
export VLLM_ASCEND_FXRT_DUMMY_QUANT=${VLLM_ASCEND_FXRT_DUMMY_QUANT:-0}
export P_DP_SIZE="$3" P_TP_SIZE="$7"
p_json() {
    python - "$1" <<'PY'
import json, os, sys
e = os.environ
def flag(k, default):
    s = e.get(k, default).lower()
    if s not in ("true", "false", "1", "0"):
        raise ValueError(k + " must be true/false/1/0")
    return s in ("true", "1")
data = {
    "compile": dict(mode=0 if e.get("P_EAGER") == "1" else 1, backend="inductor", cudagraph_mode="NONE",
                    debug_dump_path=e.get("MOE_AUDIT_DUMP", os.path.abspath("fx_dump"))),
    "additional": dict(enable_cpu_binding=flag("P_CPU_BINDING", "true"),
                       enable_shared_expert_dp=True, enable_dsa_cp=True,
                       enable_prefill_mc2=flag("P_PREFILL_MC2", "true")),
    "kv": dict(kv_connector="MooncakeHybridConnector", kv_role="kv_producer",
               kv_port=e.get("P_KV_PORT", "30000"), engine_id=e.get("P_ENGINE_ID", "0"),
               kv_connector_extra_config=dict(
                   prefill=dict(dp_size=int(e["P_DP_SIZE"]), tp_size=int(e["P_TP_SIZE"])),
                   decode=dict(dp_size=int(e.get("P_DECODE_DP", "8")), tp_size=int(e.get("P_DECODE_TP", "1"))))),
    "profiler": dict(profiler="torch", torch_profiler_dir=e.get("P_PROFILE_DIR", os.path.abspath("vllm_profile_prefill")),
                     torch_profiler_with_stack=False),
}
print(json.dumps(data[sys.argv[1]]))
PY
}
p_extra=()
entry=(vllm)
if [[ "${P_EAGER:-0}" == 1 ]]; then
    p_extra+=(--enforce-eager)
    entry=(python -m tools.moe_audit.eager_entry)
    [[ "${P_EAGER_RAW:-0}" != 1 ]] || entry=(vllm)
    export VLLM_ASCEND_ENABLE_FXRT_BACKEND=0
    export VLLM_ASCEND_ENABLE_INDUCTOR_FXRT=0
    export VLLM_ASCEND_ENABLE_INDUCTOR_ASCENDC=0
    unset VLLM_EXTERNAL_FX_BACKEND
fi
[[ -z "${P_LOAD_FORMAT:-}" ]] || p_extra+=(--load-format "$P_LOAD_FORMAT")
if [[ "${P_LOAD_FORMAT:-}" == "dummy" ]]; then
    loader_extra=()
else
    loader_extra=(--model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}')
fi
kv_extra=(--kv-transfer-config "$(p_json kv)")
if [[ "${P_DISABLE_KV:-0}" == 1 ]]; then
    kv_extra=()
fi
"${entry[@]}" serve "$P_MODEL" --host "${P_LISTEN_HOST:-0.0.0.0}" --port "$2" \
    --data-parallel-size "$3" --data-parallel-rank "$4" \
    --data-parallel-address "$5" --data-parallel-rpc-port "$6" --tensor-parallel-size "$7" \
    --enable-expert-parallel --seed 1024 --served-model-name "${P_MODEL_NAME:-dsv4}" \
    --max-model-len 4096 --max-num-batched-tokens 8192 --max-num-seqs 16 \
    --no-disable-hybrid-kv-cache-manager \
    "${loader_extra[@]}" \
    --no-enable-prefix-caching --safetensors-load-strategy prefetch \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --trust-remote-code --block-size 128 --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization "${P_MEMORY_UTIL:-0.9}" --quantization ascend \
    --compilation-config "$(p_json compile)" --additional-config "$(p_json additional)" \
    "${kv_extra[@]}" --profiler-config "$(p_json profiler)" "${p_extra[@]}"
