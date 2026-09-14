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
git switch -c run-7323-eager refs/remotes/audit7323/eager
```

依次测试完再切到另外两个分支（先停止前一个服务，确认 NPU 释放）：

```bash
git switch -c run-7323-opaque refs/remotes/audit7323/fxrt-opaque
# 或
git switch -c run-7323-split refs/remotes/audit7323/fxrt-split
```

每个分支的 `tools/moe_audit/mode` 已写好默认模式；**仅 checkout 不会自动改变原来的 P.sh**，
需要通过下面的 wrapper 启动。不要同时运行三套服务。

P.sh 使用问题中已有的现网脚本；保留：DP2、TP4、EP、DSA-CP、shared expert DP、
MTP1、max_model_len4096、max_batched_tokens8192、max_seqs16、真实权重、
CPU binding、原来的 IP/网卡及 Mooncake P2×4/D8×1 配置。
不要为了本次比较打开 enable_prefill_mc2/enable_fused_mc2，也不要强制路由。
wrapper 只替换 compile/eager 参数和诊断环境；不替换上述模型/并行参数。

## 3. 拉起并记录 prefill.log

确认所有待用卡空闲；在两个终端分别启动 DP0/DP1。下面 IP/端口请换成现网当前值，
原始 P.sh 位置也要替换；RPC 端口两边相同，HTTP 端口不同。

```bash
export VLLM_ASCEND_MOE_AUDIT_PROFILE=1
export VLLM_ASCEND_MOE_AUDIT_LIMIT=16
export MOE_AUDIT_DUMP=/workspace/fx_dump_7323/eager  # 每种模式换独立目录

# 终端一：DP0，参数顺序沿用原始 P.sh
bash tools/moe_audit/run.sh /workspace/P.sh \
  0,1,2,3 8900 2 0 7.150.1.10 13345 4 >>prefill.log 2>&1

# 终端二：同一个分支、相同的诊断环境和 P.sh
bash tools/moe_audit/run.sh /workspace/P.sh \
  4,5,6,7 8901 2 1 7.150.1.10 13345 4 >>prefill.log 2>&1
```

两终端要使用相同工作目录，保证写入同一个 prefill.log；也可使用绝对日志路径。
每种模式测试前将上一份日志改名留档，不要混合三次运行：
`mv prefill.log prefill.eager.log`（确认服务已停止后）。
不拆分/拆分模式的 JSON 仍写 `backend=inductor`，这是此基线外部 backend 入口的匹配条件；
实际经 `VLLM_EXTERNAL_FX_BACKEND=fxrt` 交给 FXRT，不是先执行 Inductor 优化。
`debug_dump_path` 直接写入 JSON，避免旧 platform 在环境默认值传播前关闭编译。

wrapper 不会替你更改原 P.sh 的 dummy quant 开关。本次若需严格复现历史配置，保持其原值；
真实权重验收不应把 dummy quant 兼容路径视作数值精度保证。请保留完整启动日志供审计。
采样打开后有 profiling 开销，**本次不用于测 TTFT 性能**；生产性能测试前关闭诊断并重启。

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
若报错，保留错误和完整 prefill.log，不必继续后续长度，也不要打印“warmup 成功”。

## 5. 过滤与回传

```bash
python tools/moe_audit/filter.py prefill.log > prefill.audit.txt
wc -lc prefill.audit.txt
```

该脚本读取所有 rank，不只保留 rank0；合并相同条件，输出实际出现的 `dp/tp/ep` rank 集合。
`OPS` 数字是 `[最小次数, 最大次数, 有该事件的采样条数]`，不是 NPU kernel 完成次数。
请分别返回三种模式的 `prefill.audit.txt`，以及请求成功/失败数。
若仍超过传输限制，先返回全部 CONFIG 和失败 STEP/ERROR，再返回 256/2048 的 STEP/OPS；
不要仅 grep 一个 rank 后据此判断所有 rank 一致。

不用 Python 时可先粗筛（不去重，可能较长）：

```bash
grep -F '[MOE_AUDIT' prefill.log > prefill.audit.raw.txt
grep -E 'async_op=True|all_to_all_single|Traceback|WorkerProc hit|Using external FX backend' \
  prefill.log | tail -60
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

## 7. 验证记录

131 A2 / dsv4-pr72-final 的本次验证结果与使用版本记录在同目录 `VALIDATION.md`。
A2 自然选择 ALLGATHER，不能用 A2 成功冒充 A3 MC2/ALLTOALL kernel 实测。
本次 A2 验证的目标是三种入口、日志及过滤脚本确实可用，A3 路由由现网回传确认。
