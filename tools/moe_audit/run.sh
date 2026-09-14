#!/usr/bin/env bash
# Wrap the user's existing P script without replacing its parallel/model settings.
set -euo pipefail
audit7323_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit7323_repo=$(cd "$audit7323_dir/../.." && pwd)
audit7323_mode=$(<"$audit7323_dir/mode")
audit7323_server=${1:?Usage: bash tools/moe_audit/run.sh /absolute/P.sh [P arguments...]}
shift
audit7323_dump=${MOE_AUDIT_DUMP:-/workspace/fx_dump_7323/$audit7323_mode}
vllm() {
    local args=() arg
    export PYTHONPATH="$audit7323_repo:${PYTHONPATH:-}"
    export VLLM_ASCEND_MOE_AUDIT=1
    export VLLM_ASCEND_MOE_AUDIT_PROFILE=${VLLM_ASCEND_MOE_AUDIT_PROFILE:-1}
    export VLLM_ASCEND_MOE_AUDIT_LIMIT=${VLLM_ASCEND_MOE_AUDIT_LIMIT:-16}
    export VLLM_USE_AOT_COMPILE=0 VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_ASCEND_ENABLE_INDUCTOR_ASCENDC=0 VLLM_ASCEND_ENABLE_INDUCTOR_FXRT=0
    unset DSV4_TEST_FORCE_MC2 DSV4_TEST_FORCE_ALLTOALL
    export VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL=0
    if [[ "$audit7323_mode" == eager ]]; then
        unset VLLM_EXTERNAL_FX_BACKEND VLLM_ASCEND_ENABLE_FXRT_BACKEND
    else
        export VLLM_EXTERNAL_FX_BACKEND=fxrt
        export VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL=$([[ "$audit7323_mode" == split ]] && echo 1 || echo 0)
        export VLLM_DEBUG_DUMP_PATH="$audit7323_dump"
        mkdir -p "$audit7323_dump"
    fi
    while (($#)); do
        arg=$1; shift
        case "$arg" in
            --compilation-config) shift ;;
            --compilation-config=*|--enforce-eager|--no-enforce-eager) ;;
            *) args+=("$arg") ;;
        esac
    done
    if [[ "$audit7323_mode" == eager ]]; then
        args+=(--enforce-eager --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}')
    else
        args+=(--compilation-config "{\"mode\":1,\"backend\":\"inductor\",\"cudagraph_mode\":\"NONE\",\"debug_dump_path\":\"$audit7323_dump\"}")
    fi
    echo "[MOE_AUDIT_VERSION] mode=$audit7323_mode ascend=$(git -C "$audit7323_repo" rev-parse HEAD) source=$audit7323_repo"
    python -c 'import importlib.metadata as m; import vllm, vllm_ascend; print("[MOE_AUDIT_VERSION]", "vllm="+vllm.__file__, "ascend="+vllm_ascend.__file__, "fxrt="+m.version("fxrt"))'
    command vllm "${args[@]}" --enable-logging-iteration-details
}
# Supports the original script's `exec vllm ...` as well as `vllm ...`.
exec() { "$@"; }
source "$audit7323_server" "$@"
