#!/usr/bin/env bash
set -euo pipefail

# Usage: start_ttft_mode.sh MODE DEVICES [LOG_ROOT]
# MODE is full (整图) or split (DSV4 prefill decomposition).
mode=${1:?full|split}
devices=${2:-"0,1;2,3"}
log_root=${3:-/workspace/dsv4/logs/ttft-$mode}
[[ "$mode" == full || "$mode" == split ]]
export PYTHONPATH=${P_ASCEND_SOURCE:-/vllm-workspace/vllm-ascend}:/vllm-workspace/vllm:${PYTHONPATH:-}
export P_NIC=${P_NIC:-lo} P_LOCAL_IP=${P_LOCAL_IP:-127.0.0.1}
export P_MODEL=${P_MODEL:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}
export P_LOAD_FORMAT=${P_LOAD_FORMAT:-dummy} P_MEMORY_UTIL=${P_MEMORY_UTIL:-0.45}
export P_PREFILL_MC2=${P_PREFILL_MC2:-true} P_KV_PORT=${P_KV_PORT:-30773}
export VLLM_ASCEND_ENABLE_FXRT_BACKEND=1
export VLLM_VERSION=0.23.0
export VLLM_ASCEND_ENABLE_INDUCTOR_FXRT=0
export VLLM_ASCEND_ENABLE_INDUCTOR_ASCENDC=0
export VLLM_ASCEND_FXRT_DUMMY_QUANT=${VLLM_ASCEND_FXRT_DUMMY_QUANT:-1}
export VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL=$([[ "$mode" == split ]] && echo 1 || echo 0)
export DSV4_TEST_MOCK_A3_ROUTE=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$log_root"
template="$(dirname "${BASH_SOURCE[0]}")/p-run_dp_template.sh"
IFS=';' read -r d0 d1 <<< "$devices"
bash "$template" "$d0" 18990 2 0 127.0.0.1 18772 2 >"$log_root/rank0.log" 2>&1 & p0=$!
bash "$template" "$d1" 18991 2 1 127.0.0.1 18772 2 >"$log_root/rank1.log" 2>&1 & p1=$!
printf '%s %s\n' "$p0" "$p1" >"$log_root/launcher.pids"
wait "$p0" "$p1"
