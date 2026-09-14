# 7.3.24：参数化 P 模板与三模式开关

本次检查范围是7.3.23交付的诊断 wrapper、模板、测试驱动。
历史日志/VALIDATION中的绝对路径是证据，不改写；旧 `server_ep8_*.sh` 保留为历史复现脚本，
仍有固定路径，不作为新的部署入口。不是宣称容器所有第三方文件均无硬编码。
新入口：本目录 `p-run_dp_template.sh` + `run.sh`。

## 配置审查的关键变化

本次提供的 P 配置新增 `enable_prefill_mc2=true`，不同于7.3.23实测的false。
在这些分支中容量公式为 `min(ceil(max_batch/TP),512)*TP`：
batch8192、TP4得到 **2048**，而非之前的32。
A3 EP8且fused=0时，selector_tokens≤2048选择MC2，大于2048选择ALLTOALL。
使用DP间最大token数，不能只按bench输入长度判断；启动profile8192也会超过容量。
开启prefill_mc2不等于开启fused_mc2；本模板不额外开启fused_mc2。
以上是源码条件推导，本次没有在A3实测此新配置。

## 模板参数

运行前先激活现有Python环境并source本机CANN环境；模板不硬编码CANN或Python安装路径。

| 参数 | 含义/默认值 |
| --- | --- |
| P_NIC（必填） | 通信网卡，如 enp23s0f3 |
| P_LOCAL_IP（必填） | HCCL使用的本机IP |
| P_MODEL（必填） | 权重目录，支持空格 |
| 位置参数1–7 | 可见卡、HTTP端口、DP数量、DP rank、DP地址、DP RPC端口、TP数量 |
| P_MODEL_NAME | 对外模型名，默认dsv4 |
| P_LISTEN_HOST | HTTP监听地址，默认0.0.0.0 |
| P_KV_PORT / P_ENGINE_ID | Mooncake端口/引擎ID，默认30000/0 |
| P_DECODE_DP / P_DECODE_TP | KV配置中的Decode拓扑，默认8/1 |
| P_PREFILL_MC2 | 默认true，复现7.3.23旧配置时显式false |
| P_CPU_BINDING | 默认true，需检查运行日志是否实际绑定成功 |
| P_MEMORY_UTIL | 默认0.9 |
| P_PROFILE_DIR | 默认当前目录/vllm_profile_prefill |
| P_LOAD_FORMAT | 默认不传，即真实权重加载；dummy仅用于裁剪测试 |
| MOE_AUDIT_DUMP | FX图目录，wrapper默认当前目录/fx_dump_7323/模式 |
| MOE_AUDIT_REPO | Ascend源码根目录，默认wrapper所在仓库 |
| MOE_AUDIT_TOOL_DIR | 包含mode文件的工具目录，默认wrapper目录 |
| MOE_AUDIT_DRY_RUN | 1时只打印最终argv和相关环境，不加载模型、不占卡 |

KV prefill DP/TP自动使用位置参数3/7，避免启动TP2而KV仍写死TP4。
原始seed、batch/seq/model_len、MTP、DSA-CP、shared expert DP、量化、loader、profiler配置保留。
**唯一主动调整的数值兼容开关：DUMMY_QUANT默认0。** 用户给出的脚本是1；
要逐项复现旧环境，请显式设置1；真实权重正确性验证建议0，不能把dummy路径当成精度保证。

## 三模式：wrapper在P脚本设置环境后再覆盖

| 项目 | eager分支 | opaque分支 | split分支 |
| --- | --- | --- | --- |
| enforce_eager | 开启 | 删除 | 删除 |
| compilation mode | 0 | 1 | 1 |
| backend | 无外部编译 | JSON写inductor，实际direct FXRT | 同左 |
| cudagraph_mode | NONE | NONE | NONE |
| VLLM_EXTERNAL_FX_BACKEND | unset | fxrt | fxrt |
| VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL | 0 | 0 | 1 |
| VLLM_DEBUG_DUMP_PATH | unset | MOE_AUDIT_DUMP | 同左 |
| VLLM_USE_AOT_COMPILE / VLLM_USE_V2_MODEL_RUNNER | 0 / 0 | 0 / 0 | 0 / 0 |
| VLLM_ASCEND_ENABLE_INDUCTOR_ASCENDC / INDUCTOR_FXRT | 0 / 0 | 0 / 0 | 0 / 0 |

两编译模式源码均保持fullgraph；此处split指内部算子展开，不是允许graph break。
`VLLM_ASCEND_ENABLE_FXRT_BACKEND`在这两个旧基线不是必要入口，不能代替external变量；
eager会unset它。wrapper删除原compilation-config并重建，不合并额外的编译优化配置。
`DSV4_TEST_FORCE_MC2/ALLTOALL`两变量统一unset，不强制路由。
统一加入日志开关AUDIT=1、PROFILE默认1、LIMIT默认16及iteration日志。
`enable_prefill_mc2`、CPU binding、MTP、EP等不随三模式自动改变，三轮比较必须保持一致。
CPU诊断profiler与原`--profiler-config`可共存，但采样时不要再调用服务的start_profile接口，
以免嵌套启动profiler；本轮不用于TTFT性能测量。

## 7.3.26现网启动

以README.md第3节为准：保留start.sh和p-launch_online_dp.py的七参数调用，
在P launcher的command中插入run.sh，D/Proxy不改动。统一输出到每轮独立目录的prefill.log。
本次现网基线P_PREFILL_MC2=true、DUMMY_QUANT=1；后者不同于参数化模板默认0，需显式设置。
原现网模板直接写定这些值时不用P_*变量；wrapper在模板export之后覆盖模式。
不使用131 mock的P_DISABLE_KV或强制路由开关，不把DP address设为0.0.0.0。

7.3.26本地无设备检查覆盖三模式×两个DP rank的最终argv/env：卡号、7100/7101、
DP2/TP4、RPC12320、KV P2×4/D8×1、prefill_mc2=true、DUMMY_QUANT=1及编译入口。
通过shell语法和参数检查不等于A3请求/profiling实测；本次未启动NPU服务。
