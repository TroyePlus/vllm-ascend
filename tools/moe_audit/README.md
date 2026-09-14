# 7.3.23：A3 八卡 Prefill 路由诊断

目的：比较同一 P 配置在 eager、FXRT 不拆分、FXRT 拆分三种入口中，为什么选择
MC2 / FUSED_MC2 / ALLTOALL / ALLGATHER。正确算子名是
`npu_moe_distribute_dispatch_v2`，不是 `MoeDistributePatchV2`。

## 1. 分支与前提

远程：<https://github.com/TroyePlus/vllm-ascend>

| 分支 | 基线 | wrapper 默认入口 |
| --- | --- | --- |
| `audit/7323-eager` | `d20ac15e7` + 已验证的 DSA-CP weight layout 适配 | enforce_eager，compile mode 0 |
| `audit/7323-fxrt-opaque` | 同上 | direct FXRT，fullgraph，不展开 DSA/MoE |
| `audit/7323-fxrt-split` | `19dcffcf6`，包含原拆分修改 | direct FXRT，fullgraph，开启 prefill 分解 |

不拆分版不是“将所有内部算子展开成一张图”：它保持原有自定义算子的边界。
拆分版也保持 `fullgraph=True`，不是通过允许 graph break 绕过问题。
三分支均不包含强制 A2 走 ALLTOALL、`async_all_to_all` 封装或最近的未提交实验修复。
共同日志位于 forward 图外，不改变选择器、输入 shape 或通信参数。
eager/opaque 的 layout 适配是明确保留的旧对照修复，不是日志修改。

依赖此前配套的 vLLM 修改：验证基线 `/vllm-workspace/vllm` 的 `f5ffef085`，
包含 external FX backend 入口；不能仅切换 Ascend 分支并使用完全原生 vLLM。
必须保留当前已能运行的 CANN、torch/torch_npu、Ascend C++ 扩展和 FXRT 安装。
这次日志修改不要求重编 C++。跨不同 csrc 基线时，先确认已有扩展兼容，不能仅凭包版本号判断。
不要在 site-packages 手改；wrapper 会把当前分支源码目录置于 PYTHONPATH 首位。

## 2. 获取分支、保留原始 P 参数

在 P 容器当前 **vllm-ascend 源码仓库**中执行：

```bash
git status --short                 # 有本地修改先保存，不要覆盖
git fetch https://github.com/TroyePlus/vllm-ascend.git \
  'refs/heads/audit/7323-*:refs/remotes/audit7323/*'
git switch -c run-7323-eager refs/remotes/audit7323/audit/7323-eager
```

依次测试完再切到另外两个分支（先停止前一个服务，确认 NPU 释放）：

```bash
git switch -c run-7323-opaque refs/remotes/audit7323/audit/7323-fxrt-opaque
# 或
git switch -c run-7323-split refs/remotes/audit7323/audit/7323-fxrt-split
```

每个分支的 `tools/moe_audit/mode` 已写好默认模式；**仅 checkout 不会自动改变原来的 P.sh**，
需要通过下面的 wrapper 启动。不要同时运行三套服务。

P.sh 使用问题中已有的现网脚本；保留：DP2、TP4、EP、DSA-CP、shared expert DP、
MTP1、max_model_len4096、max_batched_tokens8192、max_seqs16、真实权重、
CPU binding、原来的 IP/网卡及 Mooncake P2×4/D8×1 配置。
7.3.26以本次现网配置enable_prefill_mc2=true为准；旧false实验须单独标记，不混比。不要强制路由。
wrapper 只替换 compile/eager 参数和诊断环境；不替换上述模型/并行参数。

## 3. 7.3.26：按现网 launcher 启动，统一 prefill.log

保留现网 `start.sh → p-launch_online_dp.py → p-run_dp_template.sh → vllm serve` 链路，
不要套用131 mock驱动或删除KV配置。两个DP分别启动API是现网已有方式，
不能仅凭“Waiting for READY message from DP Coordinator”断定需要Decode提供READY：
该消息属于vLLM内部DP coordinator，须查各DP启动异常、地址和端口，不能与Mooncake握手混同。

### 3.1 仅修改P launcher的调用入口

在原 `p-launch_online_dp.py` 的 `run_command()` 中，把 command 的开头：

```python
command = [
    "bash",
    "./p-run_dp_template.sh",
```

改为：

```python
command = [
    "bash",
    os.path.join(os.environ["MOE_AUDIT_REPO"], "tools/moe_audit/run.sh"),
    os.path.abspath("./p-run_dp_template.sh"),
```

后面的七个参数保持原样：visible_devices、engine_port、dp_size、dp_rank、
dp_address、dp_rpc_port、tp_size。launcher已import os，无需新增依赖。
建议同时把 `dp_rpc_port = args.dp_rpc_port` 改为 `dp_rpc_port = str(args.dp_rpc_port)`，
避免省略CLI参数时默认整数传给subprocess；本例显式传12320也可正常工作。
原模板存在性检查保留。只把run.sh套在Python launcher外层不会拦截其新bash子进程，
必须在上述command中接入wrapper。

### 3.2 P模板与环境保持一致

优先使用你本次提供的现网模板，不需要复制131脚本。保留：

- P DP2/TP4/EP8，卡0–3与4–7；DSA-CP、shared expert DP、CPU binding均true。
- model_len4096、batch8192、seq16、MTP1、真实权重、多线程加载、显存比例0.9。
- **本次enable_prefill_mc2=true**；不要使用旧指导中的false。fused_mc2不额外开启。
  按本分支公式batch8192/TP4对应capacity=2048；selector实际token数≤2048通常选MC2，
  超过2048且未启用fused时选ALLTOALL。请求标称长度不等于selector token数。
- MooncakeHybridConnector、kv_producer、KV端口30000、P2×4/D8×1、engine_id=0。
- 原网卡/IP及HCCL参数；本机IP必须实际存在。DP address是可连接的master IP，
  **不能用HTTP通配监听地址0.0.0.0替代**。HTTP的--host仍为0.0.0.0。
- 为逐项复现本次脚本，保持DUMMY_QUANT=1；它是测试兼容开关，不能作为真实权重精度保证。
  三种模式都保持相同值，不在本轮同时改变数值兼容路径。

若选择仓库自带参数化模板，需要显式设置
`P_NIC=enp23s0f3 P_LOCAL_IP=7.150.1.10 P_MODEL=/data/models/DeepSeek-V4-Flash-w8a8-mtp`、
`P_PREFILL_MC2=true VLLM_ASCEND_FXRT_DUMMY_QUANT=1`；真实权重不要设置P_LOAD_FORMAT=dummy。
原现网模板使用nic_name/local_ip，不读取这些P_*变量。

wrapper在P模板export之后覆盖三模式开关，因此仅在start.sh里unset变量是不够的：

| 项目 | eager | fxrt-opaque | fxrt-split |
| --- | --- | --- | --- |
| enforce_eager / compilation mode | 开启 / 0 | 关闭 / 1 | 关闭 / 1 |
| VLLM_EXTERNAL_FX_BACKEND | unset | fxrt | fxrt |
| DECOMPOSE_DSV4_PREFILL | 0 | 0 | 1 |
| cudagraph_mode | NONE | NONE | NONE |
| JSON backend | 无外部backend | inductor（实际direct FXRT） | 同左 |

两FXRT分支保持fullgraph=True。wrapper还设置AOT_COMPILE=0、USE_V2_MODEL_RUNNER=0、
ENABLE_INDUCTOR_ASCENDC/ENABLE_INDUCTOR_FXRT=0，并清除强制MC2/ALLTOALL测试开关；
不改变DP/TP、MTP、KV或prefill_mc2。完整变量名称见PARAMETERS.md。

### 3.3 start.sh：只替换P启动行，D与Proxy保持原样

在P容器选好源码分支、停止上一轮全部P进程并确认卡已释放后，设置实际路径：

```bash
export MOE_AUDIT_REPO=/absolute/path/to/vllm-ascend
export MOE_AUDIT_TOOL_DIR="$MOE_AUDIT_REPO/tools/moe_audit"
# start.sh从原现网部署目录执行，保证./p-run_dp_template.sh存在
RUN_DIR=$(mktemp -d "$PWD/audit-7326-$(date +%Y%m%d-%H%M%S)-XXXXXX")
export MOE_AUDIT_DUMP="$RUN_DIR/fx_dump"
export VLLM_ASCEND_MOE_AUDIT_PROFILE=1
export VLLM_ASCEND_MOE_AUDIT_LIMIT=16
git -C "$MOE_AUDIT_REPO" rev-parse HEAD > "$RUN_DIR/commit.txt"
cat "$MOE_AUDIT_TOOL_DIR/mode" > "$RUN_DIR/mode.txt"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 先通过真实launcher做无卡参数检查，两个rank都必须出现
MOE_AUDIT_DRY_RUN=1 python p-launch_online_dp.py \
  --dp-size 2 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 \
  --dp-address 7.150.1.10 --dp-rpc-port 12320 --vllm-start-port 7100 \
  > "$RUN_DIR/argv-check.log" 2>&1

# 用这一行替换start.sh中的P启动行；不要保留旧P行造成重复启动
nohup python -u p-launch_online_dp.py \
  --dp-size 2 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 \
  --dp-address 7.150.1.10 --dp-rpc-port 12320 --vllm-start-port 7100 \
  > "$RUN_DIR/prefill.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/launcher.pid"
```

每轮文件名仍是无后缀的 **prefill.log**，但所在RUN_DIR唯一，避免覆盖历史。
P端口为7100/7101、DP RPC12320；D端口7200–7207、DP RPC12321；
KV30000与上述端口用途不同，不能合并。Proxy继续指向原P/D端口。
原D和Proxy命令无需改动，不要给D套P审计wrapper，也不要重复启动已经运行的D/Proxy。
切换分支前不能仅kill launcher.pid：其子进程可能继续运行，须检查该轮P API/engine/worker均退出。

### 3.4 单文件日志的含义与过滤

一次launcher的 `>prefill.log 2>&1` 打开一个文件，两个DP子进程继承输出，
**不会各自截断一次文件**；包含所有rank，不会只打印一个分支/DP。
但并发日志可能交错甚至拼接，不承诺逐行原子性。不要再起另一套launcher重定向同一路径；
三模式顺序测试、使用独立RUN_DIR。不要只grep rank0，缺失记录也不能当作算子未执行。

```bash
python "$MOE_AUDIT_TOOL_DIR/filter.py" "$RUN_DIR/prefill.log" > "$RUN_DIR/prefill.audit.txt"
grep -F '[MOE_AUDIT_VERSION]' "$RUN_DIR/prefill.log"
wc -lc "$RUN_DIR/prefill.audit.txt"
```

确认mode/ascend源码路径符合当前分支，CONFIG覆盖DP0/DP1及TP/EP rank。
未看到版本行说明wrapper未接入或启动在此前已失败。
若必须使用部署目录的./prefill.log，可保留原重定向；每轮全部P进程停止后再归档，下一轮才用>重建。
CPU审计profiler打开时不要同时调用/start_profile，避免嵌套profiler；本轮不用于TTFT测量。

## 4. 发送请求（先看 health，不吞掉 warmup 错误）

沿用现网 PD proxy，使 P/D 正常配合；不要同时使用其他压测流量。
建议先各发送两次 256 和 2048 长度请求。第二次用于观察编译缓存命中后的路径：

```bash
for n in 256 2048; do
  vllm bench serve \
    --host 7.150.1.10 --port 1999 \
    --model dsv4 --tokenizer /data/models/DeepSeek-V4-Flash-w8a8-mtp \
    --trust-remote-code --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len "$n" --random-output-len 10 \
    --ignore-eos --num-prompts 2 --max-concurrency 1 \
    2>&1 | tee "request-${n}.log"
done
```

chat 模板、benchmark 的测试请求及调度会影响实际 token 数，不能把参数 256 当成每次
forward 的实际 token 数；以日志 `selector_tokens/local/actual` 和原生 iteration 日志为准。
若报错，保留错误和完整的 `"$RUN_DIR/prefill.log"`，不必继续后续长度，也不要打印“warmup 成功”。

## 5. 过滤与回传

```bash
python tools/moe_audit/filter.py "$RUN_DIR/prefill.log" > "$RUN_DIR/prefill.audit.txt"
wc -lc "$RUN_DIR/prefill.audit.txt"
```

该脚本读取所有 rank，不只保留 rank0；合并相同条件，输出实际出现的 `dp/tp/ep` rank 集合。
`OPS` 数字是 `[最小次数, 最大次数, 有该事件的采样条数]`，不是 NPU kernel 完成次数。
请分别返回三种模式的 `prefill.audit.txt`，以及请求成功/失败数。
若仍超过传输限制，先返回全部 CONFIG 和失败 STEP/ERROR，再返回 256/2048 的 STEP/OPS；
不要仅 grep 一个 rank 后据此判断所有 rank 一致。

不用 Python 时可先粗筛（不去重，可能较长）：

```bash
grep -hF '[MOE_AUDIT' "$RUN_DIR/prefill.log" > prefill.audit.raw.txt
grep -E 'async_op=True|all_to_all_single|Traceback|WorkerProc hit|Using external FX backend' \
  "$RUN_DIR/prefill.log" | tail -60
```

## 6. 如何解释

- `CAPACITY`：记录初始化的 raw_max、max_reqs、uniform_decode_query_len、TP、512 上限。
  正式代码公式为 `min(ceil(raw_max / TP), 512) * TP`；不能永远假定容量是32。
- `CONFIG`：实际 soc、DP/TP/EP、专家数、量化、MTP、dynamic EPLB、prefill/fused 开关和编译入口。
- `SELECT`：从原选择器取得结果，`selector_tokens` 是这次调用选择器的真实参数。
  `profile`/`metadata`/`draft`/prefill/decode 字段用于区分启动32/8192、请求和 MTP；
  字段缺失为 null，不能强行解释成 decode 或 prefill。
- `END=returned`：这次 Python forward 返回；不是异步设备执行或整条 HTTP 请求成功保证。
- `ERROR`：forward 异常，原异常继续抛出，不吞错。
- `OPS`：CPU profiler 观测到的调用；首次编译可含 fake/tracing，第二次也应结合
  recompilation 日志判断。未记录某个名字不代表设备上绝对没执行该算子。
  本工具不额外 synchronize，不改变既有异步时序来证明设备完成。

关键：**SELECT=ALLTOALL 才是 MoE AllToAll 分支**。DSA-CP 自己也调用同步/functional
AllToAll；只在 FX 图或 profiler 搜到 all_to_all_single，不能证明 MoE 走了此路径。
MC2 是否调用 v2 应在 OPS 寻找 `npu_moe_distribute_dispatch_v2`；FUSED_MC2 是另一个
融合路径，不能当成 MC2 的同名开关。若拆分版在 Dynamo 处拒绝 async_op=True，
这时是入图失败，不能说 FXRT 已执行此通信算子。

源码定位：

- `vllm_ascend/ascend_forward_context.py`：`set_mc2_tokens_capacity`、
  `select_moe_comm_method`、`_select_a3_moe_comm_method`；在现网 EP8 / fused=0 下，
  `selector_tokens <= capacity` 选 MC2，否则选 ALLTOALL。
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`：MC2 dispatcher 根据
  `enable_dispatch_v2` 调用 v2 或旧 dispatch；ALLTOALL dispatcher 的
  `with_quant` 分支额外交换 scale，然后交换 hidden states。
- `vllm_ascend/ops/fused_moe/comm_utils.py`：`async_all_to_all` 真正调用
  `dist.all_to_all_single(..., async_op=True)`；本次没有将其改成同步。
- `vllm_ascend/moe_route_audit.py`：统一日志及可选 CPU profiler，MRv1 forward 图外执行。

`CONFIG.quant` 是选择器实际读取的 HF quant 字段，可能为 null；
`model_quant` 是加载的量化配置类。前者为空不能推导为模型未量化。

## 7. 验证记录

131 A2 / dsv4-pr72-final 的本次验证结果与使用版本记录在同目录 `VALIDATION.md`。
A2 自然选择 ALLGATHER，不能用 A2 成功冒充 A3 MC2/ALLTOALL kernel 实测。
本次 A2 验证的目标是三种入口、日志及过滤脚本确实可用，A3 路由由现网回传确认。
